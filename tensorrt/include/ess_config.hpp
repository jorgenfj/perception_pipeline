#pragma once

#include <string>

#include "colorize_disparity.hpp"
#include "transforms/ess_preprocess.hpp"

namespace perception {

// ESS stereo disparity parameters. The full 960x576 export only.
struct EssConfig {
  bool enabled = false;
  std::string engine_path;

  // A property of the exported engine, not a tuning knob.
  EssNormalization normalization;

  // Disparities scoring under this are drawn black and reported untrusted.
  float conf_threshold = 0.0f;

  // --- viewer: ess ----------------------------------------------------------
  DisparityColormap colormap = DisparityColormap::Turbo;

  // The display range, in network pixels. Not a constraint on the network:
  // disparities outside it clamp to the ends of the colormap.
  float display_min_disparity = 0.0f;
  float display_max_disparity = 64.0f;
};

}  // namespace perception
