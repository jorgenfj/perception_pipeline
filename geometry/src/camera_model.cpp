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

} // namespace perception::geometry
