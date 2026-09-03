#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include <perception/utils/message_types.hpp>

namespace perception {

// The device dependent types

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
