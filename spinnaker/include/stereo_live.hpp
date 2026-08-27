#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "action_sync_check.hpp"
#include "camera_config.hpp"
#include "frame_sink.hpp"

namespace perception {

// Which two cameras to open, and how many buffers each gets. Deliberately not
// stereo/stereo_config.hpp's StereoConfig: that also carries pairing tolerance,
// window geometry and a recording root, and depending on it would make the SDK
// wrapper depend on the viewer.
struct StereoCameras {
  // One per stream, in stream order. Both must be set on a real rig: empty
  // serials pick whatever is found first, and the two streams may then swap
  // between runs.
  std::string serials[2];

  // Heap buffers per stream. Must be at least what the stream mode needs.
  uint32_t buffer_count = 8;
};

// Two cameras into one callback: the live half of the stereo viewer. The SDK is
// behind the pimpl, which keeps <Spinnaker.h> out of stereo_view_main.cpp.
class LiveStereo {
 public:
  // Invoked on that stream's own acquisition thread, with the frame still in
  // the camera's buffer. The buffer is released the moment this returns, so
  // copy out of it and do not block: this is the acquisition thread, and
  // anything slow here is camera backpressure.
  using FrameCallback =
      std::function<void(uint32_t stream, const FrameMeta& meta, const void* data)>;

  // Opens both cameras and applies `camera` to each, differing only in the
  // serial from `cameras`. Throws if a serial does not match, or if the two
  // cameras disagree about geometry.
  LiveStereo(const StereoCameras& cameras, const CameraConfig& camera, FrameCallback callback);
  ~LiveStereo();

  LiveStereo(const LiveStereo&) = delete;
  LiveStereo& operator=(const LiveStereo&) = delete;

  void start();
  void stop();

  // Geometry both cameras agreed on. Checked at construction, because a stereo
  // pair whose halves are different sizes is not a stereo pair.
  const CameraGeometry& geometry() const;

  // GevIEEE1588Status per camera, e.g. {"Slave", "Slave"} once locked. A live
  // GVCP register read, so call it at startup or on the periodic report line,
  // never per frame.
  //
  // Read it and believe only the bad news. "Master" on both means neither
  // camera heard a grandmaster and each elected itself -- there is no common
  // epoch and nothing can pair. But "Slave" is not proof of sync either: it is
  // the camera's own opinion, and the offset it reports is exactly the quantity
  // that a link with asymmetric queuing delay gets wrong. Confirming sync takes
  // a scene event, not a register.
  std::vector<std::string> ptp_status();

  // One camera's PTP state, read off the node map.
  struct PtpSample {
    std::string status;        // GevIEEE1588Status: Slave / Master / Listening
    bool has_offset = false;   // false if the camera does not expose the node
    int64_t offset_ns = 0;     // GevIEEE1588OffsetFromMaster, only when Slave
  };

  // Both cameras at once. Safe to call before start(), which is the point:
  // with no acquisition running there is no second thread on the node map and
  // no image traffic on the link, so what this reads is PTP's behaviour with
  // the rig otherwise idle. Compare against the same reading under load to see
  // whether streaming is what breaks the lock.
  std::vector<PtpSample> ptp_sample();

  // Block until both cameras read PTP "Slave", or `timeout` elapses. True if
  // they got there.
  //
  // Needed because applying the camera config resets the PTP state machine:
  // both cameras come back in "Listening" and take a few Announce intervals to
  // settle, so anything that requires a lock has to wait rather than sample
  // once and give up. Prints progress, since this is a visible pause.
  bool wait_for_ptp_slave(std::chrono::milliseconds timeout);

  // The scheduled AcquisitionStart Action Command from `action_sync:`. One
  // broadcast covers both cameras. Waits up to `config.ptp_wait_ms` for the
  // lock, then throws unless both read PTP "Slave". `checker[s]` is armed to
  // verify where stream s's first frame landed.
  void arm_scheduled_start(const ActionSyncConfig& config, ActionSyncChecker& left,
                           ActionSyncChecker& right);

  // Broadcast ONE scheduled Action Command (FrameStart trigger) to both cameras,
  // to fire a single synchronized capture at PTP time `target_ns`. Unlike
  // arm_scheduled_start's one-shot AcquisitionStart, this is meant to be called
  // once per frame so every exposure -- not just the first -- is PTP-aligned,
  // which is what actually holds the two shutters together over time. Requires
  // the cameras configured for TriggerSelector=FrameStart / TriggerSource=Action0
  // (set that in camera.features before construction). Returns true if both
  // cameras acknowledged OK.
  bool send_trigger(const ActionSyncConfig& config, uint64_t target_ns);

  // True once either source has given up, i.e. no further frames can arrive.
  bool failed() const;

  // "cam0 delivered=1204 incomplete=0 timeouts=0 | cam1 ..."
  std::string health_line() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace perception
