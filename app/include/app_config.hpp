#pragma once

#include <string>

#include "camera_config.hpp"
#include "device_ring_buffer.hpp"
#include "types.hpp"
#include "upload_stage.hpp"

namespace perception {

struct PipelineConfig {
  uint32_t ingress_depth = 8;
  uint32_t device_depth = 4;
  uint32_t max_consumers = 1;
  int device_id = 0;
  ReuseWait reuse_wait = ReuseWait::DeviceWait;
  WritePolicy write_policy = WritePolicy::RoundRobin;
};

// The local debug viewer. Not part of the pipeline -- it is one more consumer
// leasing from the device ring, and switching it off removes it entirely.
struct DisplayConfig {
  bool enable = true;
  uint32_t window_width = 1280;
  uint32_t window_height = 960;
  bool vsync = false;
  double latency_scale_ms = 100.0;
};

// Verifies GigE Vision Scheduled Action Commands actually land where
// expected: schedules one Action0-triggered AcquisitionStart aligned to a
// future whole second, then checks the first `check_frames` delivered
// timestamps against that schedule. See spinnaker/README.md. Absent from the
// yaml (the normal case), `enabled` stays false and acquire runs exactly as
// before -- this does not affect ordinary free-run operation.
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
};

struct AppConfig {
  CameraConfig camera;
  PipelineConfig pipeline;
  UploadStage::Config upload;
  DisplayConfig display;
  ActionSyncConfig action_sync;
};

// Throws std::runtime_error with the offending key path on anything malformed,
// rather than silently falling back to a default. Every section is optional;
// an empty document yields the defaults above.
AppConfig load_app_config(const std::string& path);

// GenICam symbolic pixel format name -> pipeline format. Lives here rather than
// in spinnaker/ because mapping a vendor's vocabulary onto the pipeline's is
// what the composing app is for. Throws on anything unmapped.
ImageDesc to_image_desc(const CameraGeometry& geometry);

}  // namespace perception
