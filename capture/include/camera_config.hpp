#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace perception {

// A GenICam node and the value to write to it, kept as text because that is
// what a config file has. The node's own type decides how the text is parsed,
// so one list covers integers, floats, enums, booleans and commands.
using FeatureList = std::vector<std::pair<std::string, std::string>>;

struct CameraConfig {
  std::string serial;          // empty picks the first camera found
  uint64_t timeout_ms = 1000;  // GetNextImage timeout

  // Applied in order, before acquisition starts. Order is significant on real
  // cameras: ExposureAuto has to be Off before ExposureTime is writable, and
  // geometry has to be set before PayloadSize is meaningful.
  FeatureList features;

  // The same, against the transport-layer stream node map.
  FeatureList stream_features;

  // A camera error ends acquisition, not the run: the source tears the stream
  // down and re-opens it. Negative retries forever, 0 disables reconnect (the
  // first non-timeout error is then fatal, as it used to be).
  //
  // "Attempts" counts consecutive failures to get back to acquiring
  int reconnect_attempts = -1;

  // Delay before the first retry, doubled each consecutive failure up to
  // reconnect_backoff_max_ms. The first retry after a good run is immediate.
  uint64_t reconnect_backoff_ms = 500;
  uint64_t reconnect_backoff_max_ms = 5000;
};

// What the camera turned out to be once the features were applied. The pixel
// format stays a GenICam symbolic name rather than a pipeline enum -- mapping
// vendor names onto pipeline types is the composing application's job, and
// keeping it out of here is what leaves this subproject standalone.
struct CameraGeometry {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t stride_bytes = 0;
  std::string pixel_format;

  // PayloadSize: the exact per-frame transfer size, padding included.
  std::size_t frame_bytes = 0;

  // frame_bytes rounded up to the USB3 packet size, which is what a sink slot
  // must be. Sizing below this tears images on USB3 cameras.
  std::size_t buffer_bytes = 0;
};

// Reads the `camera:` section of a YAML document, ignoring everything else, so
// the same file can carry an application's own sections alongside it. Throws
// std::runtime_error naming the offending key rather than falling back to a
// default. A document with no `camera:` section yields the defaults above.
CameraConfig load_camera_config(const std::string& path);

// A Scheduled Action Command that starts acquisition on every camera in the
// group at one PTP instant, rather than whenever each happens to be told to.
// See spinnaker/README.md. Absent from the yaml (the normal case), `enabled`
// stays false and acquisition free-runs exactly as before.
struct ActionSyncConfig {
  bool enabled = false;
  uint32_t device_key = 1;
  uint32_t group_key = 1;
  uint32_t group_mask = 1;
  double expected_hz = 7.5;
  double lead_time_ms = 500.0;
  uint32_t check_frames = 60;
  double tolerance_ms = 2.0;
  // GetTimeStamp()'s latch point relative to the trigger instant is a sensor
  // convention this app has no way to know in advance -- e.g. end-of-exposure
  // instead of start-of-exposure shows up as a start offset equal to
  // ExposureTime, not a sync failure. Set this once measured so the verdict
  // checks precision, not an unrelated constant.
  double expected_start_offset_ms = 0.0;

  // How long to wait for both cameras to reach PTP "Slave" before arming.
  // Applying the camera config resets their PTP state machine, so they start in
  // "Listening" and need a few Announce intervals; sampling once and giving up
  // fails a rig that was about to work.
  uint32_t ptp_wait_ms = 20000;

  // Per-frame triggering instead of the one-shot AcquisitionStart.
  //
  // false: one scheduled Action Command aligns the *start* of Continuous
  // acquisition, then each camera free-runs at its own AcquisitionFrameRate.
  // The pair does not stay aligned: that rate comes off the sensor clock, not
  // the PTP-disciplined one, so the two crystals walk apart at their ppm
  // difference and the skew grows without bound. Fine for a short run or a
  // single camera; not for a stereo rig that has to hold sync.
  //
  // true: TriggerSelector=FrameStart / TriggerSource=Action0, and a host thread
  // sends one scheduled command per frame -- every exposure is pinned to a PTP
  // instant, so the sensor crystal only has to hold within one frame period
  // rather than across the run. This is what actually keeps two shutters
  // together. Costs host-loop timing jitter and is bounded by the camera's
  // ActionQueueSize; see trigger_lead_ms. See spinnaker/README.md.
  bool per_frame = false;

  // Trigger rate for per_frame. 0 means "use expected_hz", which keeps the one
  // rate the checker already validates against from being stated twice.
  double trigger_hz = 0.0;

  // How far ahead of its firing instant each per-frame command is scheduled.
  // Commands in flight is roughly trigger_lead_ms / (1000 / trigger_hz), and
  // exceeding the camera's ActionQueueSize comes back as OVERFLOW acks -- so
  // this has to come down as the rate goes up.
  double trigger_lead_ms = 100.0;
};

// Reads the `action_sync:` section. Shared by acquire and stereo_view so the
// two cannot drift into disagreeing about what a key means.
ActionSyncConfig load_action_sync_config(const std::string& path);

// How many buffers the standalone driver allocates when the config does not
// say. Only used by spin_acquire; the pipeline sizes its own rings.
struct StandaloneConfig {
  uint32_t buffer_count = 8;
  uint64_t max_frames = 0;  // 0 runs until interrupted

  // Write every delivered frame to a recording. Off by default: this is a
  // bring-up tool that can record, not a recorder.
  bool record = false;
  std::string record_root = "recordings";
  uint32_t staging_frames = 32;

  // The manifest's role for the single stream. "left" or "right" if this
  // camera is half of a rig being recorded one eye at a time -- it is what
  // `source.role` matches against on playback.
  std::string record_role = "mono";
};

StandaloneConfig load_standalone_config(const std::string& path);

}  // namespace perception
