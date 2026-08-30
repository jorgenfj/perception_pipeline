#include <perception/geometry/stereo.hpp>

#include <cmath>
#include <format>
#include <stdexcept>

namespace perception::geometry {

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

StereoExtrinsics StereoExtrinsics::from_row_major(
    const std::array<double, 9> &r, const std::array<double, 3> &t,
    double baseline) {
  StereoExtrinsics extrinsics;
  extrinsics.rotation << r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8];
  extrinsics.translation_m << t[0], t[1], t[2];
  extrinsics.baseline_m = baseline;
  extrinsics.validate();

  return extrinsics;
}

void StereoRectification::validate() const {
  if (eye[0].baseline_term() != 0.0) {
    throw std::runtime_error(std::format(
        "StereoRectification: eye[0] is the reference eye and must carry no "
        "baseline offset, but its P(0,3) is {}; the two cameras are listed in "
        "the wrong order",
        eye[0].baseline_term()));
  }
  if (eye[1].baseline_term() == 0.0) {
    throw std::runtime_error(
        "StereoRectification: eye[1]'s P(0,3) is zero, so the pair states no "
        "baseline; the two cameras are listed in the wrong order");
  }

  const CameraIntrinsics first = eye[0].rectified_intrinsics();
  const CameraIntrinsics second = eye[1].rectified_intrinsics();
  const auto differs = [](double a, double b) {
    return std::abs(a - b) > kIntrinsicsTolerance_px;
  };
  if (differs(first.fx, second.fx) || differs(first.fy, second.fy) ||
      differs(first.cy, second.cy)) {
    throw std::runtime_error(std::format(
        "StereoRectification: the two eyes must share fx, fy and cy, but P1 "
        "has ({}, {}, {}) and P2 has ({}, {}, {})",
        first.fx, first.fy, first.cy, second.fx, second.fy, second.cy));
  }

  if (differs(first.cx, second.cx)) {
    throw std::runtime_error(std::format(
        "StereoRectification: the eyes have principal points {} and {}, so "
        "disparity is {} at infinity rather than zero. Support for a pair with "
        "differing principal points is not implemented: depth is computed as "
        "fx * baseline / disparity, which drops that offset and reads long "
        "ranges short. Until it is, the calibration must come from a "
        "stereoRectify run with CALIB_ZERO_DISPARITY",
        first.cx, second.cx, first.cx - second.cx));
  }

  if (std::abs(disparity_to_depth(3, 3)) > kIntrinsicsTolerance_px) {
    throw std::runtime_error(std::format(
        "StereoRectification: the eyes share a principal point, so Q(3,3) must "
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

bool rectifying_rotations_agree(const StereoRectification &rectification,
                                const StereoExtrinsics &extrinsics,
                                double tolerance) {
  return (rectification.eye[1].rotation * extrinsics.rotation)
      .isApprox(rectification.eye[0].rotation, tolerance);
}

std::optional<Eigen::Vector2d> rectified_to_source_pixel(
    const CameraIntrinsics &intrinsics, const CameraDistortionModel &distortion,
    const Rectification &rectification, double u, double v) {

  const Eigen::Vector3d ray =
      rectification.rectified_intrinsics().backproject_ray({u, v});

  const Eigen::Vector3d camera_ray = rectification.rotation.transpose() * ray;

  if (camera_ray.z() <= 0.0) {
    return std::nullopt;
  }

  const Eigen::Vector2d normalized = camera_ray.hnormalized();

  const Eigen::Vector2d distorted = distortion.distort_normalized(normalized);

  return intrinsics.normalized_to_pixel(distorted);
}
} // namespace perception::geometry
