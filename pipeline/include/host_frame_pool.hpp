#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "cuda_util.hpp"
#include "host_frame.hpp"

namespace perception {

// A bounded pool of pinned host buffers, handed out as shared_ptr.
//
// This is the far side of the GPU boundary. A device ring hands out leases,
// which carry GPU-completion tracking because a slot cannot be recycled until
// every consumer's reads have RETIRED on the device -- a host-side refcount
// alone would let the producer overwrite a slot a kernel is still reading.
// Downstream of a D2H none of that applies: the bytes are on the host and
// nothing on the GPU refers to them, so plain reference counting is enough and
// a consumer needs no consumer_id, no CUDA event, and no place in the ring's
// max_consumers budget.
//
// That is the point of this type. A sink that only wants bytes -- a recorder, a
// publisher -- takes a shared_ptr<const HostFrame>, holds it for exactly as
// long as it likes, and can neither stall the producer nor corrupt a slot by
// holding it too long. The cost of that freedom is that a slow sink retains
// slots, so the pool runs dry and acquire() returns null; that is a drop, it is
// counted, and it is the same bargain the recorder already makes when its
// staging ring fills.
//
// Pinned, because these are cudaMemcpyAsync destinations: a D2H into pageable
// memory is staged through a driver bounce buffer and synchronises the stream,
// which is the one thing this design exists to avoid.
class HostFramePool {
 public:
  // `slots` buffers of `frame_bytes` each, allocated up front in one pinned
  // block: a pool that allocates per frame is a pool that stalls on the
  // allocator, and cudaMallocHost is far worse than malloc for that.
  HostFramePool(uint32_t slots, std::size_t frame_bytes, uint32_t width, uint32_t height,
                int device_id = 0);

  HostFramePool(const HostFramePool&) = delete;
  HostFramePool& operator=(const HostFramePool&) = delete;

  // A free slot, or null when every slot is still held. Null is a drop, and the
  // caller must NOT enqueue a copy it has nowhere to land -- which is why this
  // is called before the cudaMemcpyAsync rather than after it.
  //
  // The returned frame is mutable and its bytes are undefined; fill it, then
  // hand it to sinks as a shared_ptr<const HostFrame> (that conversion shares
  // the control block, so there is no second allocation and no second
  // refcount). The slot returns to the pool when the last holder drops it,
  // wherever and whenever that happens.
  std::shared_ptr<HostFrame> acquire(uint64_t timestamp_ns);

  uint32_t slots() const { return static_cast<uint32_t>(store_->frames.size()); }
  std::size_t frame_bytes() const { return store_->frame_bytes; }
  std::size_t pinned_bytes() const { return store_->frame_bytes * store_->frames.size(); }

  uint32_t in_use() const { return store_->in_use.load(std::memory_order_relaxed); }
  uint32_t peak_in_use() const { return store_->peak_in_use.load(std::memory_order_relaxed); }
  uint64_t acquired() const { return store_->acquired.load(std::memory_order_relaxed); }

  // acquire() calls that found no free slot. Non-zero means a sink is holding
  // frames longer than the pool is deep -- deepen the pool or make the sink
  // faster; the frames themselves are already gone.
  uint64_t drops() const { return store_->drops.load(std::memory_order_relaxed); }

 private:
  // The pinned block and the free list, kept behind a shared_ptr that both the
  // pool and every outstanding frame's deleter holds.
  //
  // Not an implementation detail: a sink is allowed to outlive the pipeline
  // that produced its frames -- a recorder draining its queue during shutdown
  // is exactly that -- and a deleter pointing at a destroyed pool would be a
  // use-after-free at the worst possible moment. Sharing ownership of the
  // storage makes the last holder free it, whoever that turns out to be.
  struct Store {
    ~Store();

    std::size_t frame_bytes = 0;
    int device_id = 0;
    void* block = nullptr;
    std::vector<HostFrame> frames;

    std::mutex mutex;
    std::vector<uint32_t> free;  // stack of free slot indices

    std::atomic<uint32_t> in_use{0};
    std::atomic<uint32_t> peak_in_use{0};
    std::atomic<uint64_t> acquired{0};
    std::atomic<uint64_t> drops{0};
    std::atomic<uint64_t> sequence{0};

    // Runs on whichever thread drops the last reference -- a sink's thread,
    // not the producer's.
    void release(uint32_t slot) noexcept;
  };

  std::shared_ptr<Store> store_;
};

}  // namespace perception
