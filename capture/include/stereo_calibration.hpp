#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace perception {

// A stereo rig's calibration, read back verbatim from a file an offline
// calibration produced.
//
// Nothing here computes these numbers. There is no OpenCV and no linear-algebra
// library in this build, and calibration is a fixed property of the rig rather
// than something to redo per run -- so the workflow is cv2.stereoCalibrate plus
// cv2.stereoRectify offline, dumped into the YAML that
// app/config/stereo_calibration.yaml documents. Row-major throughout, matching
// OpenCV's own dump order, so checking a value against the calibration script
// is a straight comparison rather than a transpose.
//
// Deliberately free of both CUDA and the vendor SDK: rectification is a
// statement about the cameras, and the CUDA-free viewer has as much claim on it
// as the GPU pipeline does.

struct CameraCalibration {
  // "left" / "right". Matched against the `streams:` roles so a calibration
  // cannot be silently applied to the wrong eye.
  std::string role;

  // Which camera these numbers came off. Empty if the calibration did not
  // record it; when set it is checked against the configured serial.
  std::string serial;

  // K, row-major 3x3 -- [fx 0 cx; 0 fy cy; 0 0 1], pixels, for the
  // *unrectified* image at StereoCalibration::width x height.
  std::array<double, 9> camera_matrix{};

  // Coefficients in the model's own order; plumb_bob is [k1 k2 p1 p2 k3].
  std::string distortion_model = "plumb_bob";
  std::vector<double> distortion;

  // R from stereoRectify: rotation taking this camera into the rectified
  // frame, row-major 3x3.
  std::array<double, 9> rectification_rotation{};

  // P from stereoRectify: the rectified projection, row-major 3x4. P[3] holds
  // -fx * baseline for the second camera and 0 for the first, which is where
  // the baseline enters the disparity maths.
  std::array<double, 12> rectified_projection{};

  double fx() const { return camera_matrix[0]; }
  double fy() const { return camera_matrix[4]; }
  double cx() const { return camera_matrix[2]; }
  double cy() const { return camera_matrix[5]; }
};

struct StereoCalibration {
  // Image size the calibration was solved at. Checked against the geometry the
  // cameras actually come up in -- intrinsics are in pixels, so a calibration
  // taken at a different Width/Height or a different binning is not a scaling
  // away from correct, it is wrong.
  uint32_t width = 0;
  uint32_t height = 0;

  // Indexed by stream, so camera[0] is the reference eye.
  CameraCalibration camera[2];

  // The second camera's pose in the first camera's frame: R row-major 3x3, T
  // in metres.
  std::array<double, 9> rotation{};
  std::array<double, 3> translation_m{};

  // |T|. Read from the file rather than derived, and then checked against
  // translation_m -- a file whose two statements disagree is a calibration
  // someone edited by hand, and preferring one silently is how that ships.
  double baseline_m = 0.0;

  // Q from stereoRectify, row-major 4x4: [X Y Z W]^T = Q * [u v disparity 1]^T.
  std::array<double, 16> disparity_to_depth{};

  // What the rectified projections actually imply, for consumers that want
  // depth = fx * baseline / disparity and should not have to know which corner
  // of P holds what. Derived at load; equal to baseline_m on a consistent file.
  double rectified_fx() const { return camera[0].rectified_projection[0]; }
  double rectified_baseline_m() const;

  // One line for the startup log: geometry, baseline, rectified focal length.
  std::string summary() const;
};

// Throws naming the offending key rather than falling back to a default -- an
// approximate calibration is worse than none, because it produces depth that
// looks plausible.
StereoCalibration load_stereo_calibration(const std::string& path);

}  // namespace perception
