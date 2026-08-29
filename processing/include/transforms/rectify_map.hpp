#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "stereo_calibration.hpp"
#include "transforms/rectification.hpp"

namespace perception {

// Building a rectification lookup map from a calibration. Host-only.
//
// The maths -- inverting the rectifying rotation, pushing the ray back through
// the projection, re-applying the plumb_bob distortion -- depends only on the
// calibration and the image size, so it is done once here and baked into a
// table with one entry per output pixel. The per-frame kernel in
// transforms/rectification.hpp then does no arithmetic at all: read the source
// coordinate, sample there, store. This is what cv::initUndistortRectifyMap and
// cv::remap do, split the same way.
//
// Separate from rectification.hpp because this pulls in Eigen through the
// calibration types, and nvcc has no reason to compile that.

// One entry per output pixel, row-major, packed (pitch == width). Entry (x, y)
// is the coordinate in the *source* image that rectified pixel (x, y) samples,
// with the texel-centre half pixel already folded in.
//
// `width` and `height` are the rectified output size, which is the calibrated
// size: the intrinsics are in pixels, so a map built at any other size is
// wrong, not scaled.
std::vector<float2> build_rectify_map(const CameraCalibration& camera, uint32_t width,
                                      uint32_t height, RectifyMapCoords coords);

// Build the map and upload it, in one step. The usual way in.
inline std::unique_ptr<RectifyTransform> make_rectify_transform(
    const CameraCalibration& camera, uint32_t width, uint32_t height,
    RectifyMapCoords coords = RectifyMapCoords::Normalized) {
  return std::make_unique<RectifyTransform>(build_rectify_map(camera, width, height, coords),
                                            width, height, coords);
}

}  // namespace perception
