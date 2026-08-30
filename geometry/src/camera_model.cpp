#include <perception/geometry/camera_model.hpp>

#include <cmath>
#include <format>
#include <stdexcept>

namespace perception::geometry {

Eigen::Vector2d
CameraDistortionModel::undistort_normalized(const Eigen::Vector2d &point,
                                            int iterations) const {
  if (k1 == 0.0 && k2 == 0.0 && p1 == 0.0 && p2 == 0.0 && k3 == 0.0) {
    return point;
  }

  const double convergence_tol_squared =
      kUndistortTolerance * kUndistortTolerance;

  const double x_dist = point.x();
  const double y_dist = point.y();

  double x = x_dist;
  double y = y_dist;

  for (int i = 0; i < iterations; ++i) {
    const double r2 = x * x + y * y;

    const double r_dist = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
    const double t_dist_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double t_dist_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

    const double x_next = (x_dist - t_dist_x) / r_dist;
    const double y_next = (y_dist - t_dist_y) / r_dist;

    const double dx = x_next - x;
    const double dy = y_next - y;
    const double step_sq = dx * dx + dy * dy;

    x = x_next;
    y = y_next;

    if (!std::isfinite(x) || !std::isfinite(y)) {
      break;
    }

    if (step_sq < convergence_tol_squared) {
      break;
    }
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

} // namespace perception::geometry
