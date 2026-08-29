#include <perception/geometry/camera_model.hpp>

#include <format>
#include <stdexcept>

namespace perception::geometry {

Eigen::Vector2d
CameraDistortionModel::undistort_normalized(const Eigen::Vector2d &point,
                                            int iterations) const {
  const double x_dist = point.x();
  const double y_dist = point.y();

  double x = x_dist;
  double y = y_dist;

  for (int i = 0; i < iterations; ++i) {
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;

    const double r_dist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    const double t_dist_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double t_dist_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

    x = (x_dist - t_dist_x) / r_dist;
    y = (y_dist - t_dist_y) / r_dist;

  }

  const double residual = (distort_normalized({x, y}) - point).norm();
  if (residual <= kUndistortTolerance) {
    return {x, y};
  }

  throw std::runtime_error(std::format(
      "undistort_normalized: no convergence for the distorted point "
      "({}, {}) after {} iterations -- residual {}, tolerance {}. The point is "
      "outside the region this distortion model can invert",
      x_dist, y_dist, iterations, residual, kUndistortTolerance));
}


Eigen::Vector2d rectified_to_source_pixel(
    const CameraIntrinsics &intrinsics, const CameraDistortionModel &distortion,
    const Rectification &rectification, double u, double v) {
  // Rectified pixel -> ray in the rectified frame -> ray in this camera's
  // own frame, in one matrix: (P_3x3 * R)^-1. R^-1 undoes the rectifying
  // rotation and P^-1 undoes the rectified projection.
  const Eigen::Matrix3d forward =
      rectification.projection_3x3() * rectification.rotation;

  // Well conditioned for any real calibration -- a rotation times an upper
  // triangular projection -- so a singular one means hand-edited numbers.
  const double determinant = forward.determinant();
  if (!(determinant > 1e-12 || determinant < -1e-12)) {
    throw std::runtime_error(
        std::format("rectified_to_source_pixel: P * R is singular (det = {})",
                    determinant));
  }
  const Eigen::Matrix3d inverse = forward.inverse();

  const Eigen::Vector3d ray = inverse * Eigen::Vector3d{u, v, 1.0};

  // ray.z() only vanishes for a direction parallel to the image plane, which
  // no pixel of a real projection produces. Guarded anyway so a hand-edited
  // calibration gives a finite result rather than a NaN that propagates into
  // every sample downstream.
  const double inv_w = (ray.z() != 0.0) ? 1.0 / ray.z() : 0.0;
  const Eigen::Vector2d normalized{ray.x() * inv_w, ray.y() * inv_w};

  const Eigen::Vector2d distorted = distortion.distort_normalized(normalized);

  // Back out through this camera's own K, into source pixel indices.
  //
  // The skew term is a deliberate divergence from OpenCV, whose
  // initUndistortRectifyMap applies only fx/cx and drops it. The two agree
  // exactly whenever skew is 0, which is every calibration produced with the
  // standard flags, and this way one K is applied consistently everywhere in
  // this header rather than in two subtly different ways.
  return {intrinsics.fx * distorted.x() + intrinsics.skew * distorted.y() +
              intrinsics.cx,
          intrinsics.fy * distorted.y() + intrinsics.cy};
}

} // namespace perception::geometry
