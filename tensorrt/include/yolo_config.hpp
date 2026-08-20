#pragma once

#include <cstdint>
#include <string>

namespace perception {

// YOLO object detection parameters
struct YoloConfig {
  std::string engine_path;
  uint32_t model_width = 640;
  uint32_t model_height = 640;
  float conf_threshold = 0.25f;
};

}  // namespace perception
