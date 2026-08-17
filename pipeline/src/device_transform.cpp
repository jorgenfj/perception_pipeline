#include "device_transform.hpp"

#include <stdexcept>

#include "cuda_util.hpp"

namespace perception {

void PassThroughTransform::enqueue(const void* src, const ImageDesc& input, void* dst,
                                   const ImageDesc& output, cudaStream_t stream) {
  if (input.bytes() != output.bytes()) {
    throw std::runtime_error("PassThroughTransform: input and output geometry differ");
  }
  cuda_error_check(cudaMemcpyAsync(dst, src, input.bytes(), cudaMemcpyDeviceToDevice, stream),
             "PassThroughTransform: cudaMemcpyAsync");
}

}  // namespace perception
