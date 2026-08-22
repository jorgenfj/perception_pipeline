#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace perception {


// Deliberately system_clock and not steady_clock, and deliberately the same
// clock as LatencyProbe::host_now_ns() on the pipeline side
inline uint64_t host_now_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count());
}

struct FrameMeta {
  // The camera's own clock
  uint64_t timestamp_ns = 0;

  // host_now_ns() at the instant the transport returned this frame
  uint64_t host_recv_ns = 0;

  // The camera's own frame counter. A gap in it means the frame never reached
  // the host at all, which is what distinguishes packet loss from a host-side
  // drop. Restarts across a reconnect.
  uint32_t frame_id = 0;

  // Payload length, i.e. PayloadSize -- not the slot size.
  std::size_t bytes = 0;
};

// Where a camera puts frames, expressed without naming what is on the other
// side. The source hands the sink its own pinned buffers to fill, so nothing
// here mentions CUDA and this subproject builds with no CUDA toolkit present.
//
// The lifetime contract is the whole point of the interface: a slot handed to
// commit() is off limits to the transport until consumed() says otherwise, and
// the source enforces that by holding the vendor handle for exactly that long.
class FrameSink {
 public:
  static constexpr uint32_t kNoSlot = ~0u;

  virtual ~FrameSink() = default;

  virtual uint32_t slot_count() const = 0;
  virtual std::size_t slot_bytes() const = 0;

  // The slot pointers, contiguous, for handing to a transport that fills memory
  // it did not allocate. Must stay valid for the sink's lifetime.
  virtual void* const* buffers() const = 0;

  // Which slot an address belongs to, or kNoSlot if it is not one of these
  // buffers -- which means the transport ignored them and filled its own.
  virtual uint32_t slot_of(const void* ptr) const = 0;

  // Hand over a slot the transport filled itself, in whatever order it chose.
  virtual void commit(uint32_t slot, const FrameMeta& meta) = 0;

  // True once the reader is finished and the slot may be refilled. False until
  // then, which is what stops a camera overwriting a frame still in use.
  //
  // Must be level-triggered: true for as long as the slot is free, not once on
  // the transition. A decorator composing two sinks (the recorder does exactly
  // this) reads it more than once per frame, and an edge-triggered
  // implementation would leak a slot per frame under it.
  virtual bool consumed(uint32_t slot) = 0;
};

}  // namespace perception
