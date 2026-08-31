#pragma once

#include <cstdint>
#include <memory>

#include <perception/geometry/stereo.hpp>

#include "display_config.hpp"
#include "ess_config.hpp"
#include "latency_probe.hpp"
#include "ring_pair_consumer.hpp"
#include "types.hpp"

namespace perception {

// The disparity window -- for `viewer: ess`. Unlike the other viewers this
// leases nothing itself: it drives RingPairConsumer::step() inline on its own
// thread, so the engine sees both eyes of one instant and the window shows the
// disparity computed from exactly those two frames.
//
// That thread ownership is the whole reason this class exists rather than a
// pair callback in acquire_main: a GL context is thread-current, so the window
// has to be created on the thread that presents to it.
class EssViewerConsumer {
 public:
  // `pairs` must not have been start()ed -- this drives it. `reference_stream`
  // is the same index the pair consumer anchors on, needed to tell which of the
  // two leases is calibration.cameras[0].
  EssViewerConsumer(RingPairConsumer& pairs, uint32_t reference_stream, const LatencyProbe& probe,
                    const ImageDesc& source_desc, const geometry::StereoCalibration& calibration,
                    const DisplayConfig& display_config, const EssConfig& ess_config,
                    int device_id);
  ~EssViewerConsumer();

  EssViewerConsumer(const EssViewerConsumer&) = delete;
  EssViewerConsumer& operator=(const EssViewerConsumer&) = delete;

  // A window that cannot be opened -- headless box, no X forwarding -- reports
  // and is simply not there, same as the other viewers. An engine that fails to
  // load takes the window with it: there is nothing else for it to show.
  void start();
  void stop();

  bool closed() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace perception
