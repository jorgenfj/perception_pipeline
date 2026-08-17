#pragma once

#include "device_transform.hpp"

namespace perception {

// Bilinear demosaic: any of the four 8-bit Bayer mosaics -> packed RGBA8.
//
// The cheapest reconstruction that is still colour-correct: every missing
// channel is the average of its nearest same-colour neighbours, which is a
// 3x3 neighbourhood at most. It softens edges and shows the usual zipper
// artefacts on high-contrast detail -- Malvar-He-Cutler is the next step up and
// fits the same 5x5 window -- but it is a fine baseline for detection work and
// it is what everything else gets measured against.
//
// Stateless by design: the mosaic phase comes from `input.format` on every
// call, and the geometry from the descriptors, so nothing is fixed at
// construction and one instance can be shared by any number of stages.
class DebayerTransform final : public DeviceTransform {
 public:
  // RGBA8 rather than RGB8: the extra byte buys a single 4-byte coalesced
  // store per pixel instead of three overlapping byte stores, and it is what
  // the texture and interop paths downstream expect anyway. Alpha is 255.
  ImageDesc output_desc(const ImageDesc& input) const override {
    return packed_desc(input.width, input.height, PixelFormat::RGBA8);
  }

  void enqueue(const void* src, const ImageDesc& input, void* dst, const ImageDesc& output,
               cudaStream_t stream) override;

  const char* name() const override { return "DebayerBilinear"; }
};

}  // namespace perception
