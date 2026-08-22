#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

#include "frame_sink.hpp"

namespace perception {

// A FrameSink backed by ordinary page-aligned heap memory.
//
// This is what lets the camera path run with no CUDA and no pipeline: the
// buffers are not pinned, so a GPU consumer would pay for a staging copy, but
// for bring-up, config validation, and checking that a transport accepts user
// buffers at all, none of that matters.
class HeapFrameSink final : public FrameSink {
 public:
  HeapFrameSink(uint32_t slot_count, std::size_t slot_bytes);
  ~HeapFrameSink() override;

  HeapFrameSink(const HeapFrameSink&) = delete;
  HeapFrameSink& operator=(const HeapFrameSink&) = delete;

  uint32_t slot_count() const override { return static_cast<uint32_t>(buffers_.size()); }
  std::size_t slot_bytes() const override { return slot_bytes_; }
  void* const* buffers() const override { return buffers_.data(); }
  uint32_t slot_of(const void* ptr) const override;
  void commit(uint32_t slot, const FrameMeta& meta) override;
  bool consumed(uint32_t slot) override;

  // --- reader side ---

  struct Frame {
    uint32_t slot = kNoSlot;
    const void* data = nullptr;
    FrameMeta meta;
  };

  // Oldest committed slot. Blocks up to `timeout`; false on timeout or stop.
  bool pop(Frame& out, std::chrono::milliseconds timeout);

  // Hand the slot back so the camera may refill it. Until this is called the
  // source keeps holding the vendor handle, which is what makes the buffer safe
  // to read for as long as the reader wants it.
  void release(uint32_t slot);

  void stop();

 private:
  enum class SlotState : uint8_t { Idle, InFlight };

  std::size_t slot_bytes_;
  std::vector<void*> buffers_;
  std::vector<FrameMeta> meta_;
  std::vector<SlotState> state_;

  mutable std::mutex mutex_;
  std::condition_variable frame_ready_;
  std::deque<uint32_t> ready_;
  bool running_ = true;
};

}  // namespace perception
