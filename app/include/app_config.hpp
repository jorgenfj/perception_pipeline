#pragma once

#include <string>

#include "camera_config.hpp"
#include "device_ring_buffer.hpp"
#include "display_config.hpp"
#include "recording_source.hpp"
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


struct AppConfig {
  CameraConfig camera;
  RecordingSource::Config source;
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
