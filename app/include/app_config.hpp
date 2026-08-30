#pragma once

#include <string>

#include <vector>

#include "camera_config.hpp"
#include "device_ring_buffer.hpp"
#include "display_config.hpp"
#include "recording_source.hpp"
#include "stereo_calibration.hpp"
#include "ring_pair_consumer.hpp"
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

struct StreamConfig {
  // "left" / "right" camera
  std::string role;

  // Empty picks whatever camera is found first
  std::string serial;
};

// The `stereo:` section: whether to pair at all, and how.
struct StereoConfig {
  bool enabled = false;

  // Which stream lease_latest() anchors on; the other is the one searched.
  uint32_t reference_stream = 0;

  // tolerance_ns / retry_attempts / retry_wait, straight through.
  RingPairConsumer::Config consumer;

  // Path to the calibration
  std::string calibration_path;
};

enum class ViewerMode {
  Camera,
  Yolo,
  Headless,
};

const char* to_string(ViewerMode mode);


struct AppConfig {
  CameraConfig camera;
  std::vector<StreamConfig> streams;
  RecordingSource::Config source;
  PipelineConfig pipeline;
  UploadStage::Config upload;
  StereoConfig stereo;
  ViewerMode viewer_mode = ViewerMode::Camera;
  DisplayConfig display;
  ActionSyncConfig action_sync;
  YoloConfig yolo;

  // Loaded only when stereo.enabled and stereo.calibration_path is set.
  bool have_calibration = false;
  geometry::StereoCalibration calibration;
};

AppConfig load_app_config(const std::string& path);

std::string default_config_path();

std::string resolve_next_to_exe(const std::string& path);

ImageDesc to_image_desc(const CameraGeometry& geometry);

}  // namespace perception
