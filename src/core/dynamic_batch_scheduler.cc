// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/core/dynamic_batch_scheduler.h"

#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include "src/core/constants.h"
#include "src/core/infer.h"
#include "src/core/logging.h"
#include "src/core/server_status.h"

namespace nvidia { namespace inferenceserver {

DynamicBatchScheduler::DynamicBatchScheduler(
  const ModelConfig& config, uint32_t runner_cnt, RunFunc OnSchedule)
    : scheduler_thread_cnt_(runner_cnt), idle_scheduler_thread_cnt_(0),
      pending_batch_size_(0), pending_batch_queue_cnt_(0)
{
  dynamic_batching_enabled_ = config.has_dynamic_batching();
  scheduler_threads_exit_.store(false);

  max_preferred_batch_size_ = 0;
  preferred_batch_sizes_.clear();
  pending_batch_delay_ns_ = 0;

  if (dynamic_batching_enabled_) {
    for (const auto size : config.dynamic_batching().preferred_batch_size()) {
      max_preferred_batch_size_ =
        std::max(max_preferred_batch_size_, (size_t)size);
      preferred_batch_sizes_.insert(size);
    }

    pending_batch_delay_ns_ =
      (uint64_t)config.dynamic_batching().max_queue_delay_microseconds() * 1000;
  }

  // Set nice level for scheduler threads based on model priority
  int nice = SCHEDULER_DEFAULT_NICE;
  if (config.has_optimization()) {
    switch (config.optimization().priority()) {
      case ModelOptimizationPolicy::PRIORITY_MAX:
        nice = 0;
        break;
      case ModelOptimizationPolicy::PRIORITY_MIN:
        nice = 19;
        break;
      default:
        nice = SCHEDULER_DEFAULT_NICE;
        break;
    }
  }

  // Create one scheduler thread for each requested runner. Associate
  // each scheduler thread with a runner.
  for (uint32_t c = 0; c < scheduler_thread_cnt_; ++c) {
    scheduler_threads_.emplace_back(
      new std::thread([this, c, nice]() { SchedulerThread(c, nice); }));
  }
}

DynamicBatchScheduler::~DynamicBatchScheduler()
{
  // Signal the scheduler threads to exit and then wait for them...
  {
    std::unique_lock<std::mutex> lock(mu_);
    scheduler_threads_exit_.store(true);
    cv_.notify_all();
  }

  for (auto& thd : scheduler_threads_) {
    thd->join();
  }
}

tensorflow::Status
Enqueue(
  std::shared_ptr<ModelInferStats> stats,
  std::shared_ptr<InferRequestProvider> request_provider,
  std::shared_ptr<InferResponseProvider> response_provider,
  std::function<void(tensorflow::Status)> OnComplete)
{
  return tensorflow::errors::InvalidArgument("argh");
}

void
DynamicBatchScheduler::SchedulerThread(const uint32_t runner_id, const int nice)
{
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting dynamic-batch scheduler thread " << runner_id
                   << " at nice " << nice << "...";
  } else {
    LOG_VERBOSE(1) << "Starting dynamic-batch scheduler thread " << runner_id
                   << " at default nice (requested nice " << nice
                   << " failed)...";
  }

  // For debugging delay start of threads until the queue contains the
  // specified number of entries.
  const char* dstr = getenv("TRTSERVER_DELAY_SCHEDULER");
  size_t delay_cnt = 0;
  if (dstr != nullptr) {
    delay_cnt = atoi(dstr);
    LOG_INFO << "Delaying scheduler thread " << runner_id << " until "
             << delay_cnt << " queued payloads...";
  }

  const uint64_t default_wait_microseconds = 500 * 1000;

  while (!scheduler_threads_exit_.load()) {
    std::shared_ptr<std::vector<Scheduler::Payload>> payload_batch;
    bool wake_thread = false;
    uint64_t wait_microseconds = 0;

    // Hold the lock for as short a time as possible.
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (delay_cnt > 0) {
        // Debugging... wait until queue contains 'delay_cnt' items...
        wait_microseconds = 10 * 1000;
        if (queue_.size() >= delay_cnt) {
          delay_cnt = 0;
        }
      } else if (queue_.empty()) {
        wait_microseconds = default_wait_microseconds;
      } else if (dynamic_batching_enabled_) {
        // Use dynamic batching to get request payload(s) to execute.
        wait_microseconds = GetDynamicBatch();
        if (wait_microseconds == 0) {
          for (size_t idx = 0; idx < pending_batch_queue_cnt_; ++idx) {
            payload_batch->emplace_back(queue_.front());
            queue_.pop_front();
          }

          pending_batch_size_ = 0;
          pending_batch_queue_cnt_ = 0;

          // If there are still requests in the queue after removing
          // the pending batch and if there are any idle threads then
          // wake one up to service the requests remaining in the
          // queue. We need this special wake logic for the dynamic
          // batching case because we may delay handling requests in
          // the queue and so idle the threads that would normally be
          // handling those requests. We do the actual wake outside of
          // the lock to avoid having the woken thread immediately
          // block on the lock.
          wake_thread = !queue_.empty() && (idle_scheduler_thread_cnt_ > 0);
        }
      } else {
        // No batching... execute next request payload
        payload_batch->emplace_back(queue_.front());
        queue_.pop_front();
      }

      // If no requests are to be handled, wait for notification or
      // for the specified timeout before checking the queue again.
      if (wait_microseconds > 0) {
        idle_scheduler_thread_cnt_++;
        std::chrono::microseconds wait_timeout(wait_microseconds);
        cv_.wait_for(lock, wait_timeout);
        idle_scheduler_thread_cnt_--;
      }
    }

    if (wake_thread) {
      cv_.notify_one();
    }

    if (!payload_batch->empty()) {
      auto OnCompleteQueuedPayloads =
        [payload_batch](tensorflow::Status status) {
          bool found_success = false;
          for (auto& payload : *payload_batch) {
            tensorflow::Status final_status =
              status.ok() ? (payload.status_.ok() ? payload.compute_status_
                                                  : payload.status_)
                          : status;

            // All the payloads executed together, so count 1 execution in
            // the first successful payload. Other payloads stay at 0
            // executions.
            if (!found_success && final_status.ok()) {
              payload.stats_->SetModelExecutionCount(1);
              found_success = true;
            }
            payload.complete_function_(final_status);
          }
        };

      OnSchedule_(runner_id, payload_batch.get(), OnCompleteQueuedPayloads);
    }
  }  // end runner loop

  LOG_VERBOSE(1) << "Stopping dynamic-batch scheduler thread " << runner_id
                 << "...";
}

uint64_t
DynamicBatchScheduler::GetDynamicBatch()
{
  // 'mu_' mutex must be held when this function is called. queue_
  // must not be empty.

  // Handle the cases where the pending batch or request must be
  // executed immediately.
  //
  //   1) if next request would make pending batch larger than the max
  //   preferred batch size then must execute the pending patch
  //   immediately
  //
  //   2) if no pending batch and next request on its own has batch
  //   size larger than the max preferred batch size then must execute
  //   immediately
  {
    const auto batch_size =
      queue_.front().request_provider_->RequestHeader().batch_size();
    if ((pending_batch_size_ + batch_size) >= max_preferred_batch_size_) {
      if (pending_batch_queue_cnt_ == 0) {
        pending_batch_size_ = batch_size;
        pending_batch_queue_cnt_ = 1;
      }
      return 0;
    }
  }

  // Examine the new requests. If adding these new requests to the
  // pending batch allows a preferred batch size then execute it
  // immediately. Stop examining requests if the maximum preferred
  // batch size would be exceeded.
  size_t best_preferred_batch_size = 0;
  size_t best_preferred_batch_cnt = 0;
  size_t search_batch_size = pending_batch_size_;
  size_t search_batch_cnt = pending_batch_queue_cnt_;
  for (auto idx = pending_batch_queue_cnt_; idx < queue_.size(); ++idx) {
    const auto batch_size =
      queue_[idx].request_provider_->RequestHeader().batch_size();

    if ((search_batch_size + batch_size) > max_preferred_batch_size_) {
      break;
    }

    search_batch_size += batch_size;
    search_batch_cnt++;

    if (
      preferred_batch_sizes_.find(search_batch_size) !=
      preferred_batch_sizes_.end()) {
      best_preferred_batch_size = search_batch_size;
      best_preferred_batch_cnt = search_batch_cnt;
    }
  }

  // If we found a preferred batch size then execute that.
  if (best_preferred_batch_size != 0) {
    pending_batch_size_ = best_preferred_batch_size;
    pending_batch_queue_cnt_ = best_preferred_batch_cnt;
    return 0;
  }

  pending_batch_size_ = search_batch_size;
  pending_batch_queue_cnt_ = search_batch_cnt;

  // Should always have at least one request in the pending batch at
  // this point.
  if (pending_batch_queue_cnt_ == 0) {
    LOG_ERROR << "unexpected pending batch size 0";
    return 0;
  }

  // If there is no batch queuing delay then just immediately
  // execute whatever is pending.
  if (pending_batch_delay_ns_ == 0) {
    return 0;
  }

  // Compare the age of the oldest pending request to the maximum
  // batch queuing delay and execute now if queuing delay is
  // exceeded. If queuing delay not exceeded create a timer to wakeup
  // a thread to check again at the maximum allowed delay.
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  struct timespec& queued = queue_.front().queued_timestamp_;
  uint64_t delay_ns = (now.tv_sec * NANOS_PER_SECOND + now.tv_nsec) -
                      (queued.tv_sec * NANOS_PER_SECOND + queued.tv_nsec);

  if (delay_ns >= pending_batch_delay_ns_) {
    return 0;
  }

  // Return non-zero wait microseconds to cause this thread to wait
  // until the queue delay has expired. Another thread may be awaken
  // due to incoming request to handle the pending batch before this
  // thread wakes and that is ok. But if no other request comes in
  // then this thread will wake and revist the pending batch (and at
  // that time will then see the delay has been exceeded and will send
  // the batch).
  return (pending_batch_delay_ns_ - delay_ns) / 1000;
}

}}  // namespace nvidia::inferenceserver
