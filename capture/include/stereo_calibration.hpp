#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <perception/geometry/camera_model.hpp>

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
// The numbers themselves live in the geometry types
// (perception/geometry/camera_model.hpp), which is also where the maths that
// uses them lives -- there is one representation of a camera in this build, not
// one per consumer. This header is only the file format around it.
//
// Deliberately free of CUDA and of the vendor SDK: rectification is a statement
// about the cameras, and the CUDA-free viewer has as much claim on it as the
// GPU pipeline does.

struct CameraCalibration {
  // "left" / "right". Matched against the `streams:` roles so a calibration
  // cannot be silently applied to the wrong eye.
  std::string role;

  // Which camera these numbers came off. Empty if the calibration did not
  // record it; when set it is checked against the configured serial.
  std::string serial;

  // K for the *unrectified* image at StereoCalibration::width x height.
  geometry::CameraIntrinsics intrinsics;

  // Named so the file can say which model it meant. Only "plumb_bob" parses;
  // anything else is rejected at load rather than reinterpreted.
  std::string distortion_model = "plumb_bob";
  geometry::CameraDistortionModel distortion;

  // R and P from stereoRectify. Both eyes rotate -- see the Rectification
  // docs for how the pair splits the inter-camera rotation between them.
  geometry::Rectification rectification;

  double fx() const { return intrinsics.fx; }
  double fy() const { return intrinsics.fy; }
  double cx() const { return intrinsics.cx; }
  double cy() const { return intrinsics.cy; }
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

  // The second camera's pose in the first camera's frame, translation in
  // metres.
  Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d translation_m{Eigen::Vector3d::Zero()};

  // |T|. Read from the file rather than derived, and then checked against
  // translation_m -- a file whose two statements disagree is a calibration
  // someone edited by hand, and preferring one silently is how that ships.
  double baseline_m = 0.0;

  // Q from stereoRectify: [X Y Z W]^T = Q * [u v disparity 1]^T.
  Eigen::Matrix4d disparity_to_depth{Eigen::Matrix4d::Zero()};

  // What the rectified projections actually imply, for consumers that want
  // depth = fx * baseline / disparity and should not have to know which corner
  // of P holds what. Derived at load; equal to baseline_m on a consistent file.
  double rectified_fx() const {
    return camera[0].rectification.rectified_intrinsics().fx;
  }
  double rectified_baseline_m() const;

  // One line for the startup log: geometry, baseline, rectified focal length.
  std::string summary() const;
};

// Throws naming the offending key rather than falling back to a default -- an
// approximate calibration is worse than none, because it produces depth that
// looks plausible.
StereoCalibration load_stereo_calibration(const std::string& path);

}  // namespace perception
