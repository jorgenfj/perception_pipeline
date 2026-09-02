#include <perception/utils/stereo.hpp>

#include <cmath>
#include <format>
#include <stdexcept>
#include <string>

namespace perception::utils {

void StereoExtrinsics::validate() const {
  if (!(rotation.transpose() * rotation)
           .isApprox(Eigen::Matrix3d::Identity(), 1e-9) ||
      rotation.determinant() < 0.0) {
    throw std::runtime_error(
        "StereoExtrinsics: rotation is not a proper rotation");
  }
  if (!(baseline_m > 0.0) || !std::isfinite(baseline_m)) {
    throw std::runtime_error(std::format(
        "StereoExtrinsics: baseline_m must be positive and finite, got {}",
        baseline_m));
  }

  const double norm = translation_m.norm();
  if (std::abs(norm - baseline_m) > kBaselineTolerance_m) {
    throw std::runtime_error(std::format(
        "StereoExtrinsics: baseline_m is {:.6f}m but |translation_m| is "
        "{:.6f}m; they describe the same distance and must agree",
        baseline_m, norm));
  }
}

void StereoRectification::validate() const {
  if (cameras[0].baseline_term() != 0.0) {
    throw std::runtime_error(std::format(
        "StereoRectification: cameras[0] is the reference camera and must carry no "
        "baseline offset, but its P(0,3) is {}; the two cameras are listed in "
        "the wrong order",
        cameras[0].baseline_term()));
  }
  if (cameras[1].baseline_term() == 0.0) {
    throw std::runtime_error(
        "StereoRectification: cameras[1]'s P(0,3) is zero, so the pair states no "
        "baseline; the two cameras are listed in the wrong order");
  }

  if (size.empty()) {
    throw std::runtime_error(
        "StereoRectification: size is unset. P and Q are only meaningful "
        "against the grid stereoRectify computed them for, and that size "
        "cannot be recovered from them");
  }

  const CameraIntrinsics first = cameras[0].rectified_intrinsics();
  const CameraIntrinsics second = cameras[1].rectified_intrinsics();

  if (!(first.cx >= 0.0 && first.cx < static_cast<double>(size.width) &&
        first.cy >= 0.0 && first.cy < static_cast<double>(size.height))) {
    throw std::runtime_error(std::format(
        "StereoRectification: the rectified principal point ({}, {}) is "
        "outside the {}x{} grid these projections claim to be for",
        first.cx, first.cy, size.width, size.height));
  }

  const auto differs = [](double a, double b) {
    return std::abs(a - b) > kIntrinsicsTolerance_px;
  };
  if (differs(first.fx, second.fx) || differs(first.fy, second.fy) ||
      differs(first.cy, second.cy)) {
    throw std::runtime_error(std::format(
        "StereoRectification: the two cameras must share fx, fy and cy, but P1 "
        "has ({}, {}, {}) and P2 has ({}, {}, {})",
        first.fx, first.fy, first.cy, second.fx, second.fy, second.cy));
  }

  if (differs(first.cx, second.cx)) {
    throw std::runtime_error(std::format(
        "StereoRectification: the cameras have principal points {} and {}, so "
        "disparity is {} at infinity rather than zero. Support for a pair with "
        "differing principal points is not implemented: depth is computed as "
        "fx * baseline / disparity, which drops that offset and reads long "
        "ranges short. Until it is, the calibration must come from a "
        "stereoRectify run with CALIB_ZERO_DISPARITY",
        first.cx, second.cx, first.cx - second.cx));
  }

  if (std::abs(disparity_to_depth(3, 3)) > kIntrinsicsTolerance_px) {
    throw std::runtime_error(std::format(
        "StereoRectification: the cameras share a principal point, so Q(3,3) must "
        "be zero, but Q says {}; the projections and the disparity-to-depth "
        "matrix came from different runs",
        disparity_to_depth(3, 3)));
  }
}

void StereoRectification::validate_against(const StereoExtrinsics &extrinsics,
                                           double tolerance_m) const {
  const double rectified = baseline_m();
  if (std::abs(rectified - extrinsics.baseline_m) > tolerance_m) {
    throw std::runtime_error(std::format(
        "StereoRectification: the rectified projections imply a {:.6f}m "
        "baseline but the extrinsics state {:.6f}m; rectification and "
        "extrinsics came from different calibration runs",
        rectified, extrinsics.baseline_m));
  }
}

void StereoCalibration::validate() const {
  if (size.empty()) {
    throw std::runtime_error(
        "StereoCalibration: size is unset. Intrinsics are in pixels, so a "
        "calibration with no image size cannot be checked against the images "
        "it is used on");
  }
  cameras[0].intrinsics.validate_intrinsics();
  cameras[1].intrinsics.validate_intrinsics();
  extrinsics.validate();
  rectification.validate();
  rectification.validate_against(extrinsics);
}

std::string StereoCalibration::summary() const {
  std::string line = std::format(
      "calibration: {}x{}, baseline={:.4f}m (rectified {:.4f}m), fx={:.2f}px",
      size.width, size.height, extrinsics.baseline_m, rectification.baseline_m(),
      rectification.rectified_fx());

  if (!(rectification.size == size)) {
    line += std::format(", rectified to {}x{}", rectification.size.width,
                        rectification.size.height);
  }
  return line;
}

bool rectifying_rotations_agree(const StereoRectification &rectification,
                                const StereoExtrinsics &extrinsics,
                                double tolerance) {
  return (rectification.cameras[1].rotation * extrinsics.rotation)
      .isApprox(rectification.cameras[0].rotation, tolerance);
}

} // namespace perception::utils
