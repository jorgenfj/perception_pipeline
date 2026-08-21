#pragma once

#include <Spinnaker.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "camera_config.hpp"
#include "frame_sink.hpp"

namespace perception {

// Pulls frames off one camera straight into a FrameSink's buffers.
//
// The sink's slots are handed to Spinnaker as user buffers, so the transport
// DMAs into memory the application already owns and whatever reads the frame
// reads it where it landed -- there is no staging copy.
//
// The trade is buffer lifetime. With library-owned buffers the image handle
// goes back the moment it has been copied out; here it has to be held until the
// sink says the reader is done, because releasing it lets the camera refill a
// slot still in use. The pool must therefore be deep enough to cover full read
// latency, and running dry shows up as timeouts rather than as a stall.
class SpinnakerSource {
 public:
  // Init()s the camera and applies `config.features`, so geometry is readable
  // immediately -- the sink is sized from geometry().buffer_bytes.
  SpinnakerSource(Spinnaker::CameraPtr camera, CameraConfig config);
  ~SpinnakerSource();

  SpinnakerSource(const SpinnakerSource&) = delete;
  SpinnakerSource& operator=(const SpinnakerSource&) = delete;

  // First camera whose serial matches, or the first camera at all when the
  // serial is empty. Throws if there is no match.
  static Spinnaker::CameraPtr select(Spinnaker::CameraList& cameras, const std::string& serial);

  const CameraGeometry& geometry() const { return geometry_; }

  // Breakdown of incomplete() by Spinnaker::ImageStatus, e.g.
  // "missing_packets=520 data_incomplete=40 crc=8" -- empty if incomplete() is
  // zero. This is what actually distinguishes network packet loss
  // (missing_packets, needs jumbo frames / packet delay tuning) from a real
  // CRC failure or a host-side resource problem.
  std::string incomplete_breakdown() const;

  // Minimum slot count the configured stream mode accepts as user buffers.
  uint32_t min_slot_count() const { return min_slots_; }

  // `sink` must outlive the source and have slots of at least
  // geometry().buffer_bytes. Binds its buffers as the camera's and starts
  // acquiring.
  void start(FrameSink& sink);
  void stop();

  // True only once the acquisition thread has *given up* -- reconnect is
  // disabled or its attempt budget ran out. A camera error on its own does not
  // set this: the stream is torn down and re-opened, and the run continues. The
  // thread is gone when this is set, so no further frames will ever be
  // committed; it is the signal for whoever owns the pipeline to shut down
  // rather than block forever waiting on a publish that cannot come.
  bool failed() const { return failed_.load(std::memory_order_acquire); }

  // True while the stream is down and the source is trying to get it back.
  bool reconnecting() const { return reconnecting_.load(std::memory_order_relaxed); }

  // Completed reconnects since start() -- how many times the camera dropped and
  // came back. Steadily climbing means the pipeline is only staying up because
  // of the retry loop, which is worth noticing.
  uint64_t reconnects() const { return reconnects_.load(std::memory_order_relaxed); }

  // Why it stopped -- only meaningful once failed() is true, which is also what
  // publishes this string (written before the flag, read after it).
  const std::string& failure() const { return failure_; }

  // Invoked once, on the acquisition thread, immediately after failed() is set.
  // Exists so the owner can kick whatever it is parked on -- e.g.
  // DeviceRingBuffer::wake_all() -- instead of waiting out a publish that will
  // never arrive. Must be set before start(). Anything it throws is swallowed:
  // this runs on a thread whose whole point right now is to die quietly.
  void set_failure_callback(std::function<void()> cb) { on_failure_ = std::move(cb); }

  // GevIEEE1588Status off the node map ("Slave" once locked to a master), or
  // "" if this camera doesn't expose PTP at all. A live GVCP register read,
  // not cached -- call sparingly (a startup check, or every N frames), not
  // per-frame.
  std::string ptp_status();

  // GevIEEE1588OffsetFromMaster in nanoseconds. Only meaningful once
  // ptp_status() == "Slave"; false if the node isn't readable (including:
  // camera has no PTP support at all).
  bool ptp_offset_ns(int64_t& out);

  uint64_t delivered() const { return delivered_.load(std::memory_order_relaxed); }
  uint64_t incomplete() const { return incomplete_.load(std::memory_order_relaxed); }

  // Frames whose buffer was not one of the sink's: the transport declined the
  // user buffers and fell back to its own. Non-zero means the zero-copy path is
  // not actually in effect.
  uint64_t foreign() const { return foreign_.load(std::memory_order_relaxed); }

  // GetNextImage timeouts. Under load these mean the pool ran dry because the
  // reader is holding every slot -- the backpressure signal for this path.
  uint64_t timeouts() const { return timeouts_.load(std::memory_order_relaxed); }

  // Handles held right now, and the high-water mark since start(). A peak equal
  // to the slot count means the camera had nothing left to fill at least once.
  uint32_t held() const { return held_now_.load(std::memory_order_relaxed); }
  uint32_t held_peak() const { return held_peak_.load(std::memory_order_relaxed); }

  // Grab passes entered with every slot held, so the transport could not have
  // delivered whatever it did next. Each one costs a full grab timeout.
  uint64_t starved() const { return starved_.load(std::memory_order_relaxed); }

  // Handles returned to the transport, and how long they were held. The mean is
  // over reclaimed handles only -- anything still held is not in it yet.
  uint64_t reclaimed() const { return reclaimed_.load(std::memory_order_relaxed); }
  uint64_t hold_max_us() const { return hold_max_us_.load(std::memory_order_relaxed); }
  double hold_mean_us() const {
    const uint64_t n = reclaimed_.load(std::memory_order_relaxed);
    return n ? static_cast<double>(hold_sum_us_.load(std::memory_order_relaxed)) /
                   static_cast<double>(n)
             : 0.0;
  }

 private:
  void run(FrameSink& sink);
  // Publishes `reason` then the failed() flag, and fires on_failure_ once.
  void fail(std::string reason);
  void bind_buffers(FrameSink& sink);

  // Init + features + StreamBufferCountMode + min_slots_. Split out of the
  // constructor so a reconnect re-applies exactly the same setup.
  void configure_camera();

  // Re-reads geometry and throws if it no longer matches what the sink was
  // sized for. A camera that came back with different geometry cannot be
  // reattached to buffers cut for the old one.
  void verify_geometry();

  // Bind buffers and BeginAcquisition. `hard` additionally DeInit/Init's and
  // re-applies the config first, for a camera that actually went away rather
  // than one whose stream merely errored. False (with the reason logged) if the
  // camera is not ready yet.
  bool open_stream(FrameSink& sink, bool hard);

  // EndAcquisition, drain, release. Safe to call on an already-broken stream;
  // never throws.
  void close_stream(FrameSink& sink) noexcept;

  // Reclaim in a loop until no buffer is still held by the reader, so the
  // camera cannot be re-armed onto a slot the pipeline is mid-read on. False on
  // timeout, which leaves those slots held and is why re-arming waits for it.
  bool drain_held(FrameSink& sink, std::chrono::milliseconds timeout);

  // One acquisition session. Returns empty on a requested stop, otherwise the
  // error that ended it.
  std::string grab_loop(FrameSink& sink);
  // Hand back every slot the reader has finished with. Called before each grab,
  // so the pool refills as fast as reads retire.
  void reclaim(FrameSink& sink);
  void release_held();

  Spinnaker::CameraPtr camera_;
  CameraConfig config_;
  CameraGeometry geometry_;
  uint32_t min_slots_ = 0;

  // Indexed by sink slot. A live handle here is what stops the camera reusing
  // that slot; it is dropped once the sink says the read retired.
  std::vector<Spinnaker::ImagePtr> held_;
  std::vector<std::chrono::steady_clock::time_point> held_since_;
  bool was_starved_ = false;

  std::thread thread_;
  std::atomic<bool> running_{false};

  std::string failure_;
  std::atomic<bool> failed_{false};
  std::atomic<bool> reconnecting_{false};
  std::atomic<uint64_t> reconnects_{0};
  std::function<void()> on_failure_;
  std::atomic<uint64_t> delivered_{0};
  std::atomic<uint64_t> incomplete_{0};
  std::array<std::atomic<uint64_t>, 16> incomplete_by_status_{};
  std::atomic<uint64_t> foreign_{0};
  std::atomic<uint64_t> timeouts_{0};

  std::atomic<uint32_t> held_now_{0};
  std::atomic<uint32_t> held_peak_{0};
  std::atomic<uint64_t> starved_{0};
  std::atomic<uint64_t> reclaimed_{0};
  std::atomic<uint64_t> hold_sum_us_{0};
  std::atomic<uint64_t> hold_max_us_{0};
};

}  // namespace perception
