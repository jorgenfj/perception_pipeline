#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace perception {

enum class PixelFormat : uint8_t { Bayer8_RGGB, RGB8, RGBA8, GRAY8 };
enum class MemSpace : uint8_t { HostPinned, Device };

struct FrameMeta {
  uint64_t capture_time_ns = 0;
  uint64_t sequence_id = 0;
  uint64_t tick = 0;
  uint32_t camera_id = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t stride_bytes = 0;  // row pitch, may exceed width * bpp
  PixelFormat format = PixelFormat::Bayer8_RGGB;
};

struct FrameView {
  FrameMeta meta;
  void* ptr = nullptr;
  MemSpace space = MemSpace::Device;
  cudaEvent_t ready = nullptr;
  uint32_t slot = 0;
  uint64_t slot_seq = 0;
  uint64_t slot_tick = 0;
};

}  // namespace perception
