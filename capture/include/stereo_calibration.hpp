#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <perception/utils/camera_model.hpp>
#include <perception/utils/stereo.hpp>

namespace perception {

// A stereo rig's calibration, read back verbatim from a file an offline
// calibration produced.
//
// Nothing here computes these numbers. Calibration is a fixed property of the
// rig rather than something to redo per run, so the workflow is
// cv2.stereoCalibrate plus cv2.stereoRectify offline, dumped into the YAML that
// app/config/stereo_calibration.yaml documents. The file stores everything
// row-major, matching OpenCV's own dump order, so checking a value against the
// calibration script is a straight comparison rather than a transpose.
//
// The numbers themselves live in the utils types
// (perception/utils/camera_model.hpp), which is also where the maths that
// uses them lives -- there is one representation of a camera in this build, not
// one per consumer. This header is only the file format around it.
//
// Deliberately free of CUDA and of the vendor SDK: rectification is a statement
// about the cameras, and the CUDA-free viewer has as much claim on it as the
// GPU pipeline does.

// What a camera entry says about itself: which eye it is and which physical
// camera it came off.
struct CalibrationIdentity {
  // "left" / "right", matched against the `streams:` roles so a calibration
  // cannot be silently applied to the wrong eye.
  std::string role;

  // Empty if the calibration did not record it; when set it is checked
  // against the configured serial.
  std::string serial;
};

// Throws naming the offending key rather than falling back to a default -- an
// approximate calibration is worse than none, because it produces depth that
// looks plausible.
utils::StereoCalibration load_stereo_calibration(const std::string& path);

// Reads just the `cameras[].role` and `cameras[].serial` back out of the same
// file, in stream order, for the config layer's check that this calibration
// belongs to this rig. Parses the file a second time, which costs nothing once
// at startup and keeps the identity out of the utils.
//
// Throws on the same structural problems load_stereo_calibration() does.
std::array<CalibrationIdentity, 2> read_calibration_identity(const std::string& path);

}  // namespace perception
