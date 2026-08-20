#pragma once

#include <cstdint>
#include <memory>

#include "device_ring_buffer.hpp"
#include "display_config.hpp"
#include "latency_probe.hpp"
#include "types.hpp"
#include "yolo_config.hpp"

namespace perception {

// The debug window with YOLO detections drawn on top -- for `viewer: yolo`
// (see app/include/app_config.hpp). Runs inference and display against the
// same leased frame
class YoloViewerConsumer {
 public:
  YoloViewerConsumer(DeviceRingBuffer& ring, const LatencyProbe& probe, const ImageDesc& source_desc,
                     const DisplayConfig& display_config, const YoloConfig& yolo_config,
                     uint32_t consumer_id, int device_id);
  ~YoloViewerConsumer();

  YoloViewerConsumer(const YoloViewerConsumer&) = delete;
  YoloViewerConsumer& operator=(const YoloViewerConsumer&) = delete;

  // Opens the window on its own thread and starts leasing. A window that
  // cannot be opened -- headless box, no X forwarding -- reports and is
  // simply not there, same as ViewerConsumer; an engine that fails to load
  // reports too and the window still runs, just with nothing drawn on top.
  void start();
  void stop();

  bool closed() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace perception
