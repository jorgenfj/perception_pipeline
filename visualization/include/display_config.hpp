#pragma once

#include <cstdint>

namespace perception {

// The local display window's parameters
struct DisplayConfig {
  uint32_t window_width = 1440;
  uint32_t window_height = 1080;
  bool vsync = false;
  double latency_scale_ms = 100.0;
};

}  // namespace perception
