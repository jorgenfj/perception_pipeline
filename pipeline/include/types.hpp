#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace perception {

enum class PixelFormat : uint8_t {
  Bayer8_RGGB,
  Bayer8_GRBG,
  Bayer8_GBRG,
  Bayer8_BGGR,
  GRAY8,
  RGB8,
  RGBA8,
};

// A Bayer site and a mono pixel are both one byte -- the mosaic is a layout, not
// a wider pixel -- so only the interleaved formats cost more.
constexpr std::size_t bytes_per_pixel(PixelFormat format) {
  switch (format) {
    case PixelFormat::RGB8:
      return 3;
    case PixelFormat::RGBA8:
      return 4;
    case PixelFormat::Bayer8_RGGB:
    case PixelFormat::Bayer8_GRBG:
    case PixelFormat::Bayer8_GBRG:
    case PixelFormat::Bayer8_BGGR:
    case PixelFormat::GRAY8:
      break;
  }
  return 1;
}

// Geometry of one image. `stride_bytes` is the row pitch and may exceed
// width * bytes_per_pixel -- Spinnaker pads its payload, and so would a pitched
// device allocation -- so every size calculation goes through bytes().
struct ImageDesc {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t stride_bytes = 0;
  PixelFormat format = PixelFormat::Bayer8_RGGB;

  constexpr std::size_t bytes() const {
    return static_cast<std::size_t>(height) * stride_bytes;
  }
};

// Tightly packed rows, for the sources and kernels that produce no padding.
constexpr ImageDesc packed_desc(uint32_t width, uint32_t height, PixelFormat format) {
  return ImageDesc{width, height, static_cast<uint32_t>(width * bytes_per_pixel(format)), format};
}

struct Frame {
  void* image_ptr = nullptr;
  uint64_t timestamp_ns; 
};

struct FrameView {
  Frame frame;
  cudaEvent_t data_ready_event = nullptr;
  uint32_t slot = 0;
  uint64_t slot_seq = 0;
};

struct FramePeek {
  uint64_t timestamp_ns = 0;
  uint32_t slot = 0;
  uint64_t slot_seq = 0;
};

}  // namespace perception
