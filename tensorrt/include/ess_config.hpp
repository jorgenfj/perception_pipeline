#pragma once

#include <string>

#include "colorize_disparity.hpp"
#include "transforms/ess_preprocess.hpp"

namespace perception {

// ESS stereo disparity parameters. The full 960x576 export only.
struct EssConfig {
  bool enabled = false;
  std::string engine_path;

  // The prebuilt plugin library holding the fused ops the ESS graph is built
  // from. Loaded before the engine is deserialized; empty means the engine
  // needs no plugins, which no NGC ESS export is.
  std::string plugin_path;

  // A property of the exported engine, not a tuning knob.
  EssNormalization normalization;

  // Disparities scoring under this are drawn black and reported untrusted.
  float conf_threshold = 0.0f;

  // Pinned host slots for the disparity readback, in DownloadStage. Sized in
  // whole disparity frames (960x576 float32 is 2.2MB each), and shared by every
  // consumer that wants the map on the host -- the recorder is only the first.
  // It bounds how many D2H copies may be in flight, so a slow consumer costs
  // slots rather than blocking the GPU: `download: ... drops=` climbing means
  // one held a frame longer than this is deep.
  uint32_t readback_slots = 4;

  // --- viewer: ess ----------------------------------------------------------
  DisparityColormap colormap = DisparityColormap::Turbo;

  // The display range, in network pixels. Not a constraint on the network:
  // disparities outside it clamp to the ends of the colormap.
  float display_min_disparity = 0.0f;
  float display_max_disparity = 64.0f;
};

}  // namespace perception
