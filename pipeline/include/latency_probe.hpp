#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace perception {

// Estimates the offset between the camera's clock and the host's, so that a
// frame's timestamp can be compared against host time at all.
//
// This exists because `image->GetTimeStamp()` is the camera's own clock.
// Without PTP it shares no epoch with the host at all, so `now -
// frame.timestamp_ns` is not a latency -- it is two unrelated numbers
// subtracted, and the result is garbage that looks plausible. What *is*
// recoverable without running PTP on the host
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

  // Unfiltered host-arrival-minus-camera-timestamp, before any floor is
  // subtracted. Unlike age_ns() this includes the constant part -- exposure,
  // readout, link transport -- so it moves if e.g. the link gets slower.
  // Single writer (add(), called from on_arrival on the acquisition thread);
  // snapshot_reset() is read from another thread, so treat the numbers as
  // approximate, same as LatencyWindow in acquire_main.cpp.
  struct RawWindow {
    std::atomic<int64_t> min_ns{INT64_MAX};
    std::atomic<int64_t> max_ns{INT64_MIN};
    // Nanoseconds as a double, not int64_t: this is meant to survive a
    // misbehaving clock pairing (e.g. host on steady_clock, camera on a real
    // wall-clock epoch post-PTP-lock) where a single sample can be ~1e18ns.
    // Summing 60 of those in int64_t overflows; a double sum just loses
    // precision at that scale instead of wrapping to garbage.
    std::atomic<double> sum_ns{0.0};
    std::atomic<uint64_t> count{0};

    void add(int64_t ns) {
      if (ns < min_ns.load(std::memory_order_relaxed)) min_ns.store(ns, std::memory_order_relaxed);
      if (ns > max_ns.load(std::memory_order_relaxed)) max_ns.store(ns, std::memory_order_relaxed);
      sum_ns.fetch_add(static_cast<double>(ns), std::memory_order_relaxed);
      count.fetch_add(1, std::memory_order_relaxed);
    }

    bool snapshot_reset(double& min_ms, double& mean_ms, double& max_ms) {
      const uint64_t n = count.exchange(0, std::memory_order_relaxed);
      if (n == 0) return false;
      const int64_t mn = min_ns.exchange(INT64_MAX, std::memory_order_relaxed);
      const int64_t mx = max_ns.exchange(INT64_MIN, std::memory_order_relaxed);
      const double sum = sum_ns.exchange(0.0, std::memory_order_relaxed);
      min_ms = static_cast<double>(mn) * 1e-6;
      max_ms = static_cast<double>(mx) * 1e-6;
      mean_ms = sum / static_cast<double>(n) * 1e-6;
      return true;
    }
  };

  // system_clock, not steady_clock: this has to share an epoch with whatever
  // GetTimeStamp() returns. Free-running, that's the camera's own arbitrary
  // boot-relative counter and neither clock's epoch means anything -- the
  // offset-estimation this class does is the only way to compare them. Once
  // GevIEEE1588 is enabled and locked (see spinnaker/README.md), the camera's
  // timestamp is real wall-clock time, and only system_clock (CLOCK_REALTIME)
  // shares that epoch; steady_clock does not and never will, by design.
  //
  // Trade-off: unlike steady_clock, system_clock is not guaranteed monotonic
  // -- an NTP/PTP correction can step it backward. That can transiently
  // corrupt the rolling-min floor estimate below (a step-back looks like an
  // impossibly fast frame). Rare once a clock has converged and is only
  // slewing, but don't be surprised by an occasional bogus low latency
  // reading right after (re)establishing PTP lock.
  static uint64_t host_now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count());
  }

  // Called on the acquisition thread the moment the host first sees the frame.
  // Single writer, so the read-compare-store below needs no CAS.
  void on_arrival(uint64_t camera_ts_ns) {
    const uint64_t host = host_now_ns();
    const int64_t offset = static_cast<int64_t>(host) - static_cast<int64_t>(camera_ts_ns);
    raw_.add(offset);
    arrivals_.record(camera_ts_ns, host);

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

  // Reader thread: pulls and resets the raw (unfiltered) window.
  bool raw_snapshot_reset(double& min_ms, double& mean_ms, double& max_ms) {
    return raw_.snapshot_reset(min_ms, mean_ms, max_ms);
  }

  // Nanoseconds from this frame's exact host-arrival instant (on_arrival(),
  // i.e. the moment it came out of the camera's buffer) to now -- host clock
  // on both ends, no cross-clock offset estimate involved. Unlike age_ns(),
  // this cannot be contaminated by network jitter that happened *before*
  // arrival: it is a plain diff of two timestamps taken on the same clock,
  // not camera_ts plus an estimated floor. False if the arrival record
  // already fell off the ring (frame sat unconsumed too long).
  bool processing_ns(uint64_t camera_ts_ns, int64_t& out) const {
    uint64_t host_arrival_ns = 0;
    if (!arrivals_.find(camera_ts_ns, host_arrival_ns)) return false;
    out = static_cast<int64_t>(host_now_ns()) - static_cast<int64_t>(host_arrival_ns);
    return true;
  }

 private:
  // Correlates a frame's camera timestamp with the host-clock instant
  // on_arrival() saw it, so processing_ns() can measure "buffer readout to
  // now" directly. Bounded and overwritten oldest-first rather than a map: a
  // frame a NewestOnly viewer skips while catching up should just fall off
  // the ring, not leak forever. Depth is generous headroom over the
  // pipeline's own ring depths (see app_config), not tuned to them --
  // correlation failing just means processing_ns() returns false for that
  // frame, not corruption.
  struct ArrivalRing {
    static constexpr std::size_t kSize = 64;
    mutable std::mutex mutex;
    std::array<uint64_t, kSize> camera_ts{};
    std::array<uint64_t, kSize> host_ns{};
    std::size_t next = 0;

    void record(uint64_t camera_ts_ns, uint64_t host_arrival_ns) {
      std::lock_guard<std::mutex> lock(mutex);
      camera_ts[next] = camera_ts_ns;
      host_ns[next] = host_arrival_ns;
      next = (next + 1) % kSize;
    }

    bool find(uint64_t camera_ts_ns, uint64_t& host_arrival_ns) const {
      std::lock_guard<std::mutex> lock(mutex);
      for (std::size_t i = 0; i < kSize; ++i) {
        if (camera_ts[i] == camera_ts_ns) {
          host_arrival_ns = host_ns[i];
          return true;
        }
      }
      return false;
    }
  };

  std::atomic<uint64_t> samples_{0};
  std::atomic<uint64_t> window_start_{0};
  std::atomic<int64_t> current_min_{0};
  std::atomic<int64_t> previous_min_{0};
  RawWindow raw_;
  ArrivalRing arrivals_;
};

}  // namespace perception
