#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ring_lease.hpp"
#include "types.hpp"

namespace perception {

enum class ReuseWait {
  HostSync,
  DeviceWait,
};

enum class WritePolicy {
  RoundRobin,
  ScanForFree,
};

class DeviceRingBuffer {
public:
  static constexpr uint64_t kNoTimestamp = ~0ull;

  DeviceRingBuffer(uint32_t slot_count, std::size_t slot_bytes,
                   ReuseWait reuse_wait = ReuseWait::HostSync,
                   WritePolicy write_policy = WritePolicy::RoundRobin,
                   uint32_t max_consumers = 4, int device_id = 0);
  ~DeviceRingBuffer();

  DeviceRingBuffer(const DeviceRingBuffer &) = delete;
  DeviceRingBuffer &operator=(const DeviceRingBuffer &) = delete;

  uint32_t slot_count() const { return static_cast<uint32_t>(buffers_.size()); }
  std::size_t slot_bytes() const { return slot_bytes_; }
  ReuseWait reuse_wait() const { return reuse_wait_; }
  WritePolicy write_policy() const { return write_policy_; }
  uint32_t max_consumers() const { return max_consumers_; }

  uint64_t write_stalls() const {
    return write_stalls_.load(std::memory_order_relaxed);
  }

  // --- producer ---

  WriteLease acquire_write(cudaStream_t stream);

  void *data_at_slot(uint32_t slot) { return buffers_[slot]; }

  // --- consumer ---

  ReadLease lease_latest(uint32_t consumer_id, cudaStream_t stream,
                         uint32_t max_attempts = 4);

  // Lease the frame nearest `timestamp_ns` within `tol`.
  // `closest_match` false returns the first slot inside the tolerance.
  ReadLease lease_by_timestamp(uint64_t timestamp_ns, uint64_t tol,
                               uint32_t consumer_id, cudaStream_t stream,
                               bool closest_match, uint32_t max_attempts = 4);

  // --- unleased inspection ---
  //
  // These read slot metadata without holding anything, so the producer may
  // recycle the slot at any moment.
  bool view_latest_inplace(FramePeek &out) const;
  bool get_view_by_timestamp(uint64_t timestamp_ns, uint64_t tol,
                             FramePeek &out, bool closest_match) const;
  bool read_was_clean(const FramePeek &view) const;

  // --- consumer wakeup ---
  uint64_t wait_seq() const { return wait_seq_.load(std::memory_order_acquire); }
  void wait_for_publish(uint64_t seen) const {
    wait_seq_.wait(seen, std::memory_order_acquire);
  }

  // Release every waiter so it can re-check and, typically, exit. Safe to call
  // repeatedly and from a thread that publishes nothing.
  void wake_all();

private:
  friend class ReadLease;
  friend class WriteLease;

  static constexpr uint32_t kNoSlot = ~0u;

  void cleanup() noexcept;

  void wait_unleased(uint32_t slot);
  uint32_t claim_free_slot();
  void order_behind_readers(uint32_t slot, cudaStream_t stream);

  void publish_slot(uint32_t slot, uint64_t timestamp_ns, cudaStream_t stream);
  void abandon_slot(uint32_t slot) noexcept;

  cudaError_t drop_slot_hold(uint32_t slot, uint32_t consumer_id,
                             cudaStream_t stream) noexcept;

  ReuseWait reuse_wait_;
  WritePolicy write_policy_;
  uint32_t max_consumers_;
  std::size_t slot_bytes_;
  std::vector<void *> buffers_;
  std::vector<cudaEvent_t> data_ready_event_;
  std::vector<cudaEvent_t> read_done_event_;
  std::vector<std::atomic<uint64_t>> seq_;
  std::vector<std::atomic<uint64_t>> timestamp_ns_;
  std::vector<std::atomic<uint32_t>> readers_;

  uint32_t next_write_slot_ = 0;
  std::atomic<uint32_t> latest_{kNoSlot};
  std::atomic<uint64_t> write_stalls_{0};
  std::atomic<uint64_t> wait_seq_{0};
};

} // namespace perception
