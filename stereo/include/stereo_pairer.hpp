#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace perception {

// Live stereo pairing on the host, for the two cameras this project drives.
//
// Same question as the offline merge in frame_pairing.hpp, same tolerance, same
// precondition (strictly under half a frame period) -- so the two agree on
// which frames belong together. They can differ on *yield*, because live cannot
// see the future and a partner that never arrives has to be given up on.
//
// Where the CUDA path pairs by asking the other device ring for a timestamp
// (pipeline/include/ring_pair_consumer.hpp), this pairs by holding a few frames of
// each stream on the host and matching heads. It exists because the viewer and
// the recorder have to work on a machine with no GPU in it.
//
// Frames are copied in. The camera slot is released the moment push() returns,
// so a slow viewer cannot become camera backpressure -- the same reasoning that
// makes the recorder copy out, see recording_plan.md.
class StereoPairer {
 public:
  struct Config {
    uint64_t tolerance_ns = 500'000;

    // Frames held per stream. This is the hold window: a partner still in
    // flight has this many frame periods to turn up. Oldest is dropped when
    // full, which is what bounds the memory.
    uint32_t queue_frames = 8;

    // How long a frame with no partner queued waits before it is released
    // unpaired. A host clock, and only ever for giving up -- never for deciding
    // what pairs with what, which is camera timestamps and nothing else.
    //
    // Without it, one dead camera would blank the window instead of showing
    // the live one next to a stale frame, which is the opposite of what you
    // want from the tool you reach for when a camera has died.
    std::chrono::milliseconds hold{40};
  };

  // What try_pop() hands back. Pointers are into the pairer's own buffers and
  // stay valid until the next try_pop() on the same object.
  struct Pair {
    uint64_t pair_id = 0;

    // A single-sided result is a frame that could not pair: either its partner
    // never arrived within Config::hold, or the other stream has already moved
    // past this instant. Every frame pushed comes out exactly once, paired or
    // single -- an unpairable frame is still shown, because a rig that is not
    // pairing is exactly when you need to see what both cameras are doing.
    bool have[2] = {false, false};

    uint64_t timestamp_ns[2] = {0, 0};
    uint64_t host_recv_ns[2] = {0, 0};
    uint32_t frame_id[2] = {0, 0};
    const unsigned char* data[2] = {nullptr, nullptr};
    std::size_t bytes[2] = {0, 0};

    // timestamp_ns[1] - timestamp_ns[0]. Only meaningful when both sides are
    // present.
    int64_t skew_ns = 0;

    bool complete() const { return have[0] && have[1]; }
  };

  explicit StereoPairer(const Config& config);

  StereoPairer(const StereoPairer&) = delete;
  StereoPairer& operator=(const StereoPairer&) = delete;

  // Copy one frame in. Called on each camera's own acquisition thread; the two
  // streams never contend with each other, only with the consumer.
  // False means the queue was full and the oldest frame was dropped to make
  // room -- the consumer is not keeping up, which costs it frames and nothing
  // else.
  bool push(uint32_t stream, uint64_t timestamp_ns, uint64_t host_recv_ns, uint32_t frame_id,
            const void* data, std::size_t bytes);

  // Next pair, if there is one. False means neither stream has a matchable head
  // yet. Single-consumer.
  bool try_pop(Pair& out);

  uint64_t paired() const { return paired_; }
  uint64_t unpaired(uint32_t stream) const { return unpaired_[stream]; }
  uint64_t overrun(uint32_t stream) const {
    return overrun_[stream].load(std::memory_order_relaxed);
  }
  int64_t max_abs_skew_ns() const { return max_abs_skew_ns_; }

  // e.g. "stereo paired=1204 unpaired=0/2 dropped=0/0 max_skew=41us"
  std::string health_line() const;

 private:
  struct Entry {
    uint64_t timestamp_ns = 0;
    uint64_t host_recv_ns = 0;
    uint32_t frame_id = 0;
    std::vector<unsigned char> data;
    std::size_t bytes = 0;
    std::chrono::steady_clock::time_point arrived;
  };

  // Moves `entry` out of its queue into the held slot, so the buffer it points
  // at survives exactly until the next try_pop().
  void hold_entry(uint32_t stream, Entry&& entry, Pair& out);

  Config config_;

  // One lock per stream rather than one for both: the two acquisition threads
  // then never wait on each other, and try_pop() takes both together.
  mutable std::mutex mutex_[2];
  std::deque<Entry> queue_[2];
  std::vector<std::vector<unsigned char>> recycled_[2];

  // The buffers behind the last result, kept alive for the caller.
  Entry held_[2];

  // Consumer-thread only, so plain -- except overrun_, which the producers
  // bump and the consumer reports.
  uint64_t next_pair_id_ = 0;
  uint64_t paired_ = 0;
  uint64_t unpaired_[2] = {0, 0};
  std::atomic<uint64_t> overrun_[2];
  int64_t max_abs_skew_ns_ = 0;
};

}  // namespace perception
