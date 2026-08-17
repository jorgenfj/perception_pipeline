#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <stop_token>
#include <vector>

namespace perception {

// How the pinned slots get filled.
enum class FillMode {
  // The producer copies into the slot: acquire() -> host_ptr() -> commit().
  Staged,
  // A vendor DMA writes into the slot directly and picks the slot itself, so
  // there is nothing to acquire and the fill order is not ours to choose.
  External,
};

// Blocks the aquisition on backlog since camera buffers is newest only.
class HostIngressRing {
 public:
  static constexpr uint32_t kNoSlot = ~0u;

  HostIngressRing(uint32_t slot_count, std::size_t slot_bytes, int device_id = 0,
                  FillMode fill = FillMode::Staged);
  ~HostIngressRing();

  HostIngressRing(const HostIngressRing&) = delete;
  HostIngressRing& operator=(const HostIngressRing&) = delete;

  uint32_t slot_count() const { return static_cast<uint32_t>(host_.size()); }
  std::size_t slot_bytes() const { return slot_bytes_; }
  FillMode fill_mode() const { return fill_; }

  // --- producer (Staged) ---
  uint32_t acquire();

  void* host_ptr(uint32_t slot);

  // Hand the filled slot to the consumer. `bytes` is what was actually written,
  // which may be less than slot_bytes().
  void commit(uint32_t slot, uint64_t timestamp_ns, std::size_t bytes);

  // Give an acquired slot back unfilled -- the source timed out, or the frame
  // was incomplete. The consumer never saw it, so the same slot comes back out
  // of the next acquire(). Must be the most recently acquired slot.
  void abandon(uint32_t slot);

  // --- producer (External) ---

  // The slot pointers, in the layout Spinnaker's
  // SetUserBuffers(void**, count, size) wants. Stable for the ring's lifetime.
  void* const* buffers() const { return host_.data(); }

  // Which slot a vendor buffer address belongs to, or kNoSlot if the address is
  // not one of ours -- which means the transport ignored the user buffers and
  // is filling its own, so the frame cannot be taken zero-copy.
  uint32_t slot_of(const void* ptr) const;

  // Commit a slot the vendor filled itself. Any slot, in any order: the
  // transport picks, not us.
  void commit_external(uint32_t slot, uint64_t timestamp_ns, std::size_t bytes);

  // True once the consumer has released the slot and its H2D has retired, so
  // the vendor handle behind it can go back to the pool. False until then --
  // holding that handle is the only thing stopping the camera from refilling a
  // slot the GPU is still reading.
  bool slot_consumed(uint32_t slot);

  // --- consumer ---

  struct Staged {
    uint32_t slot = kNoSlot;
    const void* data = nullptr;
    std::size_t bytes = 0;
    uint64_t timestamp_ns = 0;
  };

  // Oldest committed slot. Blocks up to `timeout`; false on timeout or stop.
  // For callers driving the ring inline, where the timeout is the only thing
  // that hands control back.
  bool pop(Staged& out, std::chrono::milliseconds timeout);

  // The same, but with no deadline: sleeps until a frame is committed, the ring
  // stops, or `stoken` is asked to stop.
  bool pop(Staged& out, std::stop_token stoken);

  // Return the slot to the producer. `stream` must be the stream the H2D
  // reading this slot was enqueued on -- the event recorded here is what
  // acquire() waits on, so releasing against the wrong stream reintroduces the
  // torn-upload race this ring exists to close. Call it as soon as the copy is
  // enqueued; there is no reason to hold the slot across the transform.
  void release(uint32_t slot, cudaStream_t stream);

  // --- lifecycle ---

  // Wakes both sides so they can exit. acquire() returns kNoSlot and pop()
  // returns false from here on.
  void stop();
  bool running() const;

  uint64_t committed() const;
  uint64_t consumed() const;

 private:
  // Frees whatever has been allocated so far. Idempotent, so it serves both the
  // destructor and a constructor that throws halfway.
  void release_all() noexcept;

  bool take_ready(Staged& out);

  // Where a slot is in the round trip. Only External needs to track this: the
  // producer has to tell "handed to the consumer" from "back in the pool", and
  // in Staged mode the acquired_/released_ counters already say that.
  enum class SlotState : uint8_t { Idle, Committed, Releasing };

  std::size_t slot_bytes_;
  FillMode fill_;
  std::vector<void*> host_;
  std::vector<cudaEvent_t> h2d_done_;
  std::vector<uint64_t> timestamp_ns_;
  std::vector<std::size_t> bytes_;
  std::vector<SlotState> state_;

  mutable std::mutex mutex_;
  std::condition_variable slot_free_;  // producer waits
  std::condition_variable_any frame_ready_;  // consumer waits

  // Committed and waiting to be popped, oldest first. Explicit rather than
  // derived from a counter, because External commits in whatever order the
  // transport chose.
  std::deque<uint32_t> ready_;

  // Monotonic, never wrapped; in Staged mode the slot index is the counter
  // modulo depth, and a slot is free to acquire when acquired_ - released_ <
  // depth.
  uint64_t acquired_ = 0;
  uint64_t committed_ = 0;
  uint64_t released_ = 0;
  bool running_ = true;
};

}  // namespace perception
