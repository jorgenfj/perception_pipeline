#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "types.hpp"

namespace perception {

// Latest-wins: the producer never blocks on a consumer, so a slow consumer has
// its slot reused underneath it. still_valid() is how it finds out. For long
// running consumers, use snapshot_latest() to copy out of the ring buffer.
class DeviceRingBuffer {
 public:
  DeviceRingBuffer(uint32_t slot_count, std::size_t slot_bytes, int device_id = 0);
  ~DeviceRingBuffer();

  DeviceRingBuffer(const DeviceRingBuffer&) = delete;
  DeviceRingBuffer& operator=(const DeviceRingBuffer&) = delete;

  uint32_t slot_count() const { return static_cast<uint32_t>(buffers_.size()); }
  std::size_t slot_bytes() const { return slot_bytes_; }

  // --- producer ---

  // Next writable slot, round-robin. Blocks until the previous work that wrote
  // this slot has finished, so a kernel can never scribble over a buffer still
  // being produced into.
  uint32_t acquire_write_slot();

  // Where the producing kernel writes. Valid until this slot is acquired again.
  void* data_at_slot(uint32_t slot);

  // Record readiness against `stream` and hand the slot to consumers. The caller
  // must already have enqueued its production on that stream. Returns
  // immediately -- nothing here waits on the GPU.
  void mark_written(uint32_t slot, const FrameMeta& meta, cudaStream_t stream);

  // --- consumer ---

  // Newest published slot; false if nothing has been published yet.
  bool fetch_latest_slot(FrameView& out) const;

  // False once the producer has reused the slot this view points at, meaning
  // the data was being overwritten while the consumer worked on it.
  bool read_was_clean(const FrameView& view) const;

  // Copy the newest slot into consumer-owned memory and confirm it was not
  // reclaimed mid-copy, retrying if it was.
  bool snapshot_latest(FrameView& out, void* dst, cudaStream_t stream, cudaEvent_t copied,
                       uint32_t max_attempts = 3) const;

 private:
  static constexpr uint32_t kNoSlot = ~0u;

  // Frees whatever has been allocated so far. Idempotent, so it serves both the
  // destructor and a constructor that throws halfway.
  void release() noexcept;

  std::size_t slot_bytes_;
  std::vector<void*> buffers_;
  std::vector<cudaEvent_t> ready_;
  std::vector<FrameMeta> meta_;
  std::vector<std::atomic<uint64_t>> generation_;
  uint32_t next_write_slot_ = 0;  // producer-only, needs no synchronisation
  std::atomic<uint32_t> latest_{kNoSlot};
};

}  // namespace perception
