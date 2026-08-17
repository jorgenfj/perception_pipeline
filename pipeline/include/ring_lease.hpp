#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace perception {

class DeviceRingBuffer;



// A held read slot. While it lives, the producer cannot recycle the slot, so the
// data under it is stable rather than merely unlikely to be torn.
class ReadLease {
 public:
  ReadLease() = default;
  ~ReadLease();

  ReadLease(ReadLease&& other) noexcept;
  ReadLease& operator=(ReadLease&& other) noexcept;
  ReadLease(const ReadLease&) = delete;
  ReadLease& operator=(const ReadLease&) = delete;

  bool valid() const { return ring_ != nullptr; }

  uint32_t slot() const { return slot_; }
  uint64_t seq() const { return seq_; }
  uint64_t timestamp_ns() const { return timestamp_ns_; }
  const void* data() const { return data_; }
  cudaEvent_t data_ready_event() const { return data_ready_event_; }

  // Records read completion against the lease's stream and drops the hold. Call
  // it once the reading work is *enqueued* -- there is no need to wait for it.
  void drop_hold();

 private:
  friend class DeviceRingBuffer;
  ReadLease(DeviceRingBuffer* ring, uint32_t slot, uint64_t seq, uint32_t consumer_id,
            cudaStream_t stream, uint64_t timestamp_ns, const void* data,
            cudaEvent_t data_ready_event)
      : ring_(ring),
        slot_(slot),
        seq_(seq),
        consumer_id_(consumer_id),
        stream_(stream),
        timestamp_ns_(timestamp_ns),
        data_(data),
        data_ready_event_(data_ready_event) {}

  DeviceRingBuffer* ring_ = nullptr;
  uint32_t slot_ = 0;
  uint64_t seq_ = 0;
  uint32_t consumer_id_ = 0;
  cudaStream_t stream_ = nullptr;
  uint64_t timestamp_ns_ = 0;
  const void* data_ = nullptr;
  cudaEvent_t data_ready_event_ = nullptr;
};

// A held write slot. The slot's seq is odd for the lease's lifetime, so no
// consumer can select it. publish() brings it back to even and hands it over;
class WriteLease {
 public:
  WriteLease() = default;
  ~WriteLease();

  WriteLease(WriteLease&& other) noexcept;
  WriteLease& operator=(WriteLease&& other) noexcept;
  WriteLease(const WriteLease&) = delete;
  WriteLease& operator=(const WriteLease&) = delete;

  bool valid() const { return ring_ != nullptr; }

  uint32_t slot() const { return slot_; }
  void* data() const { return data_; }

  // Records readiness against the lease's stream and publishes. The caller must
  // already have enqueued its production there. Nothing here waits on the GPU.
  void publish(uint64_t timestamp_ns);

 private:
  friend class DeviceRingBuffer;
  WriteLease(DeviceRingBuffer* ring, uint32_t slot, cudaStream_t stream, void* data)
      : ring_(ring), slot_(slot), stream_(stream), data_(data) {}

  DeviceRingBuffer* ring_ = nullptr;
  uint32_t slot_ = 0;
  cudaStream_t stream_ = nullptr;
  void* data_ = nullptr;
};

}  // namespace perception
