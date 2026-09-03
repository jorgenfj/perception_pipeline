#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>

namespace perception {


// Deliberately system_clock and not steady_clock, and deliberately the same
// clock as LatencyProbe::host_now_ns()
inline uint64_t host_now_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count());
}

// TAI - UTC in seconds, as the kernel currently holds it. 37 today; 0 means
// nothing has set it, i.e. no phc2sys/ptp4l is disciplining this host's clock
// yet -- see ptp_timebase_ready().
inline int64_t tai_offset_s() {
  timespec tai{};
  timespec utc{};
  clock_gettime(CLOCK_TAI, &tai);
  clock_gettime(CLOCK_REALTIME, &utc);
  // Rounded from the nanosecond difference rather than differencing tv_sec:
  // the two reads straddle a second boundary now and then.
  const int64_t delta_ns =
      (static_cast<int64_t>(tai.tv_sec) - static_cast<int64_t>(utc.tv_sec)) * 1'000'000'000ll +
      (static_cast<int64_t>(tai.tv_nsec) - static_cast<int64_t>(utc.tv_nsec));
  return (delta_ns + 500'000'000ll) / 1'000'000'000ll;
}

// Now, in the timebase a camera evaluates a Scheduled Action Command against.
//
// NOT the same clock as host_now_ns(), and this is the whole point: PTP counts
// TAI, CLOCK_REALTIME counts UTC, and the two run 37 seconds apart today (more
// after the next leap second). A target built from host_now_ns() therefore
// lands ~37 s in the camera's past, every command comes back ACTION_LATE, and
// the camera fires it immediately on receipt instead -- so the rig looks
// triggered while nothing is actually scheduled, which is exactly the failure
// this function exists to prevent. Anything compared against GetTimeStamp() on
// a PTP-locked camera belongs in this timebase too.
//
// Use host_now_ns() for host-side wall-clock work -- frame arrival, latency
// against host_recv_ns -- where UTC is the correct and intended epoch.
inline uint64_t ptp_now_ns() {
  timespec ts{};
  clock_gettime(CLOCK_TAI, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ull +
         static_cast<uint64_t>(ts.tv_nsec);
}

// False when the kernel holds no TAI offset, which makes ptp_now_ns() a second
// name for host_now_ns() and silently reintroduces the 37-second error. Worth
// saying out loud before arming triggers rather than debugging it later.
inline bool ptp_timebase_ready() { return tai_offset_s() != 0; }

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
