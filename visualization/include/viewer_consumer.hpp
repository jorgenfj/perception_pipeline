#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

#include "device_ring_buffer.hpp"
#include "display_config.hpp"
#include "latency_probe.hpp"
#include "types.hpp"

namespace perception {

// The local debug window, as a self-contained consumer of the device ring --
// frames only, nothing drawn on top. For `viewer: yolo` (see
// app/include/app_config.hpp) see yolo_viewer_consumer.hpp instead; the two
// do not share a base class, since a shared one would need a boxes-or-not
// branch inside the one thing this class is meant to keep simple.
//
// It is not part of the pipeline. It leases on its own consumer id and its own
// CUDA stream, so falling behind costs it frames and nothing upstream notices,
// and switching it off removes it rather than leaving a branch on the data
// path.
//
// Nothing GL appears in this header -- not GLFW, not OpenGL, not GlViewer --
// and there is a do-nothing implementation for builds without
// PERCEPTION_WITH_DISPLAY. That is what lets the composition root name this
// type unconditionally and stay free of display #ifdefs.
class ViewerConsumer {
 public:
  // `ring` and `probe` must outlive this. `consumer_id` has to be within the
  // ring's max_consumers and must not collide with the pipeline consumer's.
  ViewerConsumer(DeviceRingBuffer& ring, const LatencyProbe& probe, const ImageDesc& desc,
                 const DisplayConfig& config, uint32_t consumer_id, int device_id);
  ~ViewerConsumer();

  ViewerConsumer(const ViewerConsumer&) = delete;
  ViewerConsumer& operator=(const ViewerConsumer&) = delete;

  // Opens the window on its own thread and starts leasing. A window that cannot
  // be opened -- headless box, no X forwarding -- reports and is simply not
  // there; the pipeline is the point and this is an inspector, so that is not
  // an error and does not count as closed().
  void start();

  // Asks the thread to leave and joins it. Idempotent, and safe when start()
  // was never called or the window never opened.
  void stop();

  // True once a window that really did open has gone away, which in practice
  // means the user closed it. False when the display is disabled, compiled out,
  // or failed to open -- in those builds nothing here should end the run, and
  // only a signal does.
  //
  // The thread calls ring.wake_all() on its way out, so a consumer parked in
  // wait_for_publish notices this promptly rather than at the next frame.
  bool closed() const { return closed_.load(std::memory_order_relaxed); }

 private:
  void run();

  DeviceRingBuffer* ring_;
  const LatencyProbe* probe_;
  ImageDesc desc_;
  DisplayConfig config_;
  uint32_t consumer_id_;
  int device_id_;

  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> closed_{false};
};

}  // namespace perception
