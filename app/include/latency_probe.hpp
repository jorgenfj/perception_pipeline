#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>

namespace perception {

// Estimates the offset between the camera's clock and the host's, so that a
// frame's timestamp can be compared against host time at all.
//
// This exists because `image->GetTimeStamp()` is the camera's own clock. It
// shares no epoch with `steady_clock`, so `now - frame.timestamp_ns` is not a
// latency -- it is two unrelated numbers subtracted, and the result is garbage
// that looks plausible. What *is* recoverable without running PTP on the host
// is the offset, estimated as the minimum of (host arrival - camera stamp) over
// a window: the frame that spent the least time getting here carries the least
// queueing, so its delta is the closest thing to the pure offset.
//
// READ THE RESULT CORRECTLY. Because the minimum is subtracted away, the number
// this yields is *latency above the fastest frame observed*, not glass-to-glass.
// The constant floor -- exposure, sensor readout, link transport -- is invisible
// from here by construction. What this does catch, exactly, is jitter, queueing,
// and every millisecond the pipeline itself adds. For a true glass-to-glass
// figure you still need a millisecond stopwatch in frame.
//
// The window exists because the two crystals drift against each other; a
// lifetime minimum would decay into a bound that no longer holds. Two buckets
// rolled every kWindowNs approximate a rolling minimum for two atomics.
class LatencyProbe {
 public:
  static constexpr uint64_t kWindowNs = 5'000'000'000ull;

  static uint64_t host_now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
  }

  // Called on the acquisition thread the moment the host first sees the frame.
  // Single writer, so the read-compare-store below needs no CAS.
  void on_arrival(uint64_t camera_ts_ns) {
    const uint64_t host = host_now_ns();
    const int64_t offset = static_cast<int64_t>(host) - static_cast<int64_t>(camera_ts_ns);

    if (samples_.fetch_add(1, std::memory_order_relaxed) == 0) {
      window_start_.store(host, std::memory_order_relaxed);
      current_min_.store(offset, std::memory_order_relaxed);
      previous_min_.store(offset, std::memory_order_relaxed);
      return;
    }

    if (host - window_start_.load(std::memory_order_relaxed) > kWindowNs) {
      previous_min_.store(current_min_.load(std::memory_order_relaxed), std::memory_order_relaxed);
      current_min_.store(offset, std::memory_order_relaxed);
      window_start_.store(host, std::memory_order_relaxed);
    } else if (offset < current_min_.load(std::memory_order_relaxed)) {
      current_min_.store(offset, std::memory_order_relaxed);
    }
  }

  bool offset_ns(int64_t& out) const {
    if (samples_.load(std::memory_order_relaxed) == 0) return false;
    out = std::min(current_min_.load(std::memory_order_relaxed),
                   previous_min_.load(std::memory_order_relaxed));
    return true;
  }

  // Host-clock instant this frame was captured, as best it can be known.
  bool capture_host_ns(uint64_t camera_ts_ns, uint64_t& out) const {
    int64_t offset = 0;
    if (!offset_ns(offset)) return false;
    out = static_cast<uint64_t>(static_cast<int64_t>(camera_ts_ns) + offset);
    return true;
  }

  // Nanoseconds between capture and now, on one clock. Negative is possible for
  // a frame that beats the current offset estimate -- that frame *is* the new
  // minimum, and the next call will reflect it.
  bool age_ns(uint64_t camera_ts_ns, int64_t& out) const {
    uint64_t captured = 0;
    if (!capture_host_ns(camera_ts_ns, captured)) return false;
    out = static_cast<int64_t>(host_now_ns()) - static_cast<int64_t>(captured);
    return true;
  }

  uint64_t samples() const { return samples_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint64_t> samples_{0};
  std::atomic<uint64_t> window_start_{0};
  std::atomic<int64_t> current_min_{0};
  std::atomic<int64_t> previous_min_{0};
};

}  // namespace perception
