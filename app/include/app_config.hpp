#pragma once

#include <string>

#include "camera_config.hpp"
#include "device_ring_buffer.hpp"
#include "display_config.hpp"
#include "types.hpp"
#include "upload_stage.hpp"
#include "yolo_config.hpp"

namespace perception {

struct PipelineConfig {
  uint32_t ingress_depth = 8;
  uint32_t device_depth = 4;
  uint32_t max_consumers = 1;
  int device_id = 0;
  ReuseWait reuse_wait = ReuseWait::DeviceWait;
  WritePolicy write_policy = WritePolicy::RoundRobin;
};

enum class ViewerMode {
  Camera,
  Yolo,
  Headless,
};

const char* to_string(ViewerMode mode);

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
  ViewerMode viewer_mode = ViewerMode::Camera;
  DisplayConfig display;
  ActionSyncConfig action_sync;
  YoloConfig yolo;
};

AppConfig load_app_config(const std::string& path);

std::string default_config_path();

std::string resolve_next_to_exe(const std::string& path);

ImageDesc to_image_desc(const CameraGeometry& geometry);

}  // namespace perception
