#pragma once

#include <cuda_runtime.h>

#include "types.hpp"

namespace perception {

class DeviceTransform {
 public:
  virtual ~DeviceTransform() = default;

  virtual ImageDesc output_desc(const ImageDesc& input) const = 0;

  virtual void prepare(const ImageDesc& input, cudaStream_t stream) {
    (void)input;
    (void)stream;
  }

  virtual void enqueue(const void* src, const ImageDesc& input, void* dst,
                       const ImageDesc& output, cudaStream_t stream) = 0;

  virtual const char* name() const = 0;
};

// Passthrough transform. Still pays the extra D2D copy DeviceTransform does.
class PassThroughTransform final : public DeviceTransform {
 public:
  ImageDesc output_desc(const ImageDesc& input) const override { return input; }

  void enqueue(const void* src, const ImageDesc& input, void* dst, const ImageDesc& output,
               cudaStream_t stream) override;

  const char* name() const override { return "PassThrough"; }
};

}  // namespace perception
