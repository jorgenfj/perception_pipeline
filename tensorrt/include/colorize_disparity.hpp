#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace perception {

// Which rainbow the disparity is painted with. Both run dark blue at
// `min_disparity` (far) through cyan, green and yellow to dark red at
// `max_disparity` (near), which is the look Isaac ROS's ESS samples have.
enum class DisparityColormap {
  // Google's Turbo: the same shape as Jet without its banding and its
  // washed-out cyan/yellow bands. The better one to read a depth edge off.
  Turbo,
  // MATLAB/OpenCV COLORMAP_JET, for matching a reference image exactly.
  Jet,
};

// Disparity in network pixels -> RGBA8 in `surface`, one texel per disparity
// pixel, so `surface` is sized width x height (e.g. the GL interop array
// GlViewer::present_gpu maps). `min_disparity`..`max_disparity` is the display
// range, not a constraint on the network: values outside it clamp to the ends
// of the colormap.
//
// Pixels with no usable disparity are left black, the way ROS's disparity_view
// leaves unmatched ones: a non-positive disparity, and -- when `confidence` is
// non-null -- one scoring under `conf_threshold`.
//
// Nothing here touches host memory.
void colorize_disparity(const float* disparity, const float* confidence, uint32_t width,
                        uint32_t height, float min_disparity, float max_disparity,
                        float conf_threshold, DisparityColormap colormap,
                        cudaSurfaceObject_t surface, cudaStream_t stream);

}  // namespace perception
