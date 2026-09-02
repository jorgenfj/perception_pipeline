#include <perception/utils/camera_model.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

namespace perception::utils {

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

namespace {

void require_sizes(ImageSize source, ImageSize target, const char *what) {
  if (source.empty()) {
    throw std::runtime_error(std::format("{}: zero-sized source image", what));
  }
  if (target.empty()) {
    throw std::runtime_error(std::format("{}: zero-sized target image", what));
  }
}

} // namespace

CameraIntrinsics scale_intrinsics(const CameraIntrinsics &intrinsics,
                                  double scale) {
  if (!(scale > 0.0) || !std::isfinite(scale)) {
    throw std::runtime_error(std::format(
        "scale_intrinsics: scale must be positive and finite, got {}", scale));
  }

  return CameraIntrinsics{
      .fx = scale * intrinsics.fx,
      .fy = scale * intrinsics.fy,
      .cx = scale * (intrinsics.cx + 0.5) - 0.5,
      .cy = scale * (intrinsics.cy + 0.5) - 0.5,
  };
}

CameraIntrinsics crop_intrinsics(const CameraIntrinsics &intrinsics,
                                 double left, double top) {
  if (!std::isfinite(left) || !std::isfinite(top)) {
    throw std::runtime_error(std::format(
        "crop_intrinsics: crop must be finite, got ({}, {})", left, top));
  }

  CameraIntrinsics cropped = intrinsics;
  cropped.cx -= left;
  cropped.cy -= top;
  return cropped;
}

double resize_scale(ImageSize source, ImageSize target, ResizeFit fit) {
  require_sizes(source, target, "resize_scale");

  const double by_width = static_cast<double>(target.width) / source.width;
  const double by_height = static_cast<double>(target.height) / source.height;

  return fit == ResizeFit::Crop ? std::max(by_width, by_height)
                                : std::min(by_width, by_height);
}

Eigen::Vector2d resize_offset(ImageSize source, ImageSize target,
                              ResizeFit fit) {
  const double scale = resize_scale(source, target, fit);

  return {0.5 * (source.width * scale - target.width),
          0.5 * (source.height * scale - target.height)};
}

CameraIntrinsics resize_intrinsics(const CameraIntrinsics &intrinsics,
                                   ImageSize source, ImageSize target,
                                   ResizeFit fit) {
  const Eigen::Vector2d offset = resize_offset(source, target, fit);

  return crop_intrinsics(
      scale_intrinsics(intrinsics, resize_scale(source, target, fit)),
      offset.x(), offset.y());
}

} // namespace perception::utils
