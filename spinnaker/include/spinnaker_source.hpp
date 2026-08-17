#pragma once

#include <Spinnaker.h>

#include <atomic>
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

  // Minimum slot count the configured stream mode accepts as user buffers.
  uint32_t min_slot_count() const { return min_slots_; }

  // `sink` must outlive the source and have slots of at least
  // geometry().buffer_bytes. Binds its buffers as the camera's and starts
  // acquiring.
  void start(FrameSink& sink);
  void stop();

  uint64_t delivered() const { return delivered_.load(std::memory_order_relaxed); }
  uint64_t incomplete() const { return incomplete_.load(std::memory_order_relaxed); }

  // Frames whose buffer was not one of the sink's: the transport declined the
  // user buffers and fell back to its own. Non-zero means the zero-copy path is
  // not actually in effect.
  uint64_t foreign() const { return foreign_.load(std::memory_order_relaxed); }

  // GetNextImage timeouts. Under load these mean the pool ran dry because the
  // reader is holding every slot -- the backpressure signal for this path.
  uint64_t timeouts() const { return timeouts_.load(std::memory_order_relaxed); }

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

  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> delivered_{0};
  std::atomic<uint64_t> incomplete_{0};
  std::atomic<uint64_t> foreign_{0};
  std::atomic<uint64_t> timeouts_{0};
};

}  // namespace perception
