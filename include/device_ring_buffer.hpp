#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "types.hpp"

namespace perception {

// Sync policy for aquire_write_slot
enum class ReuseWait {
  HostSync,
  DeviceWait,
};

// Latest-wins: the producer never blocks on a consumer, so a slow consumer has
// its slot reused underneath it. still_valid() is how it finds out. For long
// running consumers, use snapshot_latest() to copy out of the ring buffer.
class DeviceRingBuffer {
 public:
  DeviceRingBuffer(uint32_t slot_count, std::size_t slot_bytes,
                   ReuseWait reuse_wait = ReuseWait::HostSync, int device_id = 0);
  ~DeviceRingBuffer();

  DeviceRingBuffer(const DeviceRingBuffer&) = delete;
  DeviceRingBuffer& operator=(const DeviceRingBuffer&) = delete;

  uint32_t slot_count() const { return static_cast<uint32_t>(buffers_.size()); }
  std::size_t slot_bytes() const { return slot_bytes_; }
  ReuseWait reuse_wait() const { return reuse_wait_; }

  // --- producer ---

  // Next writable slot, round-robin. Waits for the previous work that wrote this
  // slot to finish, so a kernel can never scribble over a buffer still being
  // produced into -- on the calling thread under HostSync, on `stream` under
  // DeviceWait.
  uint32_t acquire_write_slot(cudaStream_t stream);

  // Where the producing kernel writes. Valid until this slot is acquired again.
  void* data_at_slot(uint32_t slot);

  // Record readiness against `stream` and hand the slot to consumers. The caller
  // must already have enqueued its production on that stream. Returns
  // immediately -- nothing here waits on the GPU.
  void mark_slot_written(uint32_t slot, uint64_t timestamp_ns, cudaStream_t stream);

  // --- consumer ---

  // Newest published slot; false if nothing has been published yet.
  bool view_latest_inplace(FrameView& out) const;

  // Retrieve FrameView by matching timestamp within tolerance
  // best_match, if false return first match within tol
  bool get_view_by_timestamp(uint64_t tick, uint64_t tol, FrameView& out, bool best_match) const;

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

  ReuseWait reuse_wait_;
  std::size_t slot_bytes_;
  std::vector<void*> buffers_;
  std::vector<cudaEvent_t> data_ready_event_;
  std::vector<cudaStream_t> write_stream_;
  std::vector<std::atomic<uint64_t>> seq_;
  std::vector<std::atomic<uint64_t>> timestamp_ns_;
  uint32_t next_write_slot_ = 0;  // producer-only, needs no synchronisation
  std::atomic<uint32_t> latest_{kNoSlot};
};

}  // namespace perception
