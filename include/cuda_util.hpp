#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace perception {

inline void cuda_error_check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

// The legacy default stream implicitly synchronises with every blocking stream
// in the process, including ones created inside libraries we do not control.
// Nothing in this pipeline may use it, so the ring APIs reject it outright
// rather than let it quietly serialise the whole GPU path.
inline void reject_default_stream(cudaStream_t stream, const char* op) {
  if (stream == nullptr || stream == cudaStreamLegacy) {
    throw std::runtime_error(std::string(op) +
                             ": refusing the legacy default stream, which serialises against "
                             "unrelated work; use a perception::CudaStream");
  }
}

// Non-blocking by construction. There is deliberately no way to get a blocking
// stream out of this type.
class CudaStream {
 public:
  CudaStream() {
    cuda_error_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
  }
  ~CudaStream() {
    if (stream_) cudaStreamDestroy(stream_);
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  cudaStream_t get() const { return stream_; }
  operator cudaStream_t() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

// Timing disabled by default: these are waited on, never measured.
class CudaEvent {
 public:
  explicit CudaEvent(unsigned flags = cudaEventDisableTiming) {
    cuda_error_check(cudaEventCreateWithFlags(&event_, flags), "cudaEventCreateWithFlags");
  }
  ~CudaEvent() {
    if (event_) cudaEventDestroy(event_);
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  cudaEvent_t get() const { return event_; }
  operator cudaEvent_t() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

}  // namespace perception
