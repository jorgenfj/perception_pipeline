#pragma once

#include <Spinnaker.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
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
  void bind_buffers(FrameSink& sink);
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
