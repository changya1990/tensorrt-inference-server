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
#pragma once

#include "tensorflow/core/lib/core/errors.h"

namespace nvidia { namespace inferenceserver {

class InferRequestProvider;
class InferResponseProvider;
class ModelInferStats;

// Scheduler interface.
class Scheduler {
 public:
  virtual ~Scheduler() {}

  // The data associated with each request being scheduled.
  struct Payload {
    Payload() = default;
    Payload(const Payload& payload) = default;
    Payload(
      struct timespec queued_timestamp, std::shared_ptr<ModelInferStats> stats,
      std::shared_ptr<InferRequestProvider> request_provider,
      std::shared_ptr<InferResponseProvider> response_provider,
      std::function<void(tensorflow::Status)> complete_function)
        : queued_timestamp_(queued_timestamp), stats_(stats),
          request_provider_(request_provider),
          response_provider_(response_provider),
          complete_function_(complete_function),
          status_(tensorflow::Status::OK()),
          compute_status_(tensorflow::Status::OK())
    {
    }

    struct timespec queued_timestamp_;
    std::shared_ptr<ModelInferStats> stats_;
    std::shared_ptr<InferRequestProvider> request_provider_;
    std::shared_ptr<InferResponseProvider> response_provider_;
    std::function<void(tensorflow::Status)> complete_function_;
    tensorflow::Status status_;
    tensorflow::Status compute_status_;
  };

  // Enqueue a request with the scheduler.
  virtual tensorflow::Status Enqueue(
    std::shared_ptr<ModelInferStats> stats,
    std::shared_ptr<InferRequestProvider> request_provider,
    std::shared_ptr<InferResponseProvider> response_provider,
    std::function<void(tensorflow::Status)> OnComplete) = 0;
};

}}  // namespace nvidia::inferenceserver
