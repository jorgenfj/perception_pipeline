#include <perception/utils/stereo.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

namespace perception::utils {

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

std::vector<float> build_rectify_map(const PinholeCameraModel &camera,
                                     const Rectification &rectification,
                                     ImageSize rectified_size) {
  if (rectified_size.empty()) {
    throw std::runtime_error("build_rectify_map: zero-sized rectified image");
  }
  const uint32_t width = rectified_size.width;
  const uint32_t height = rectified_size.height;

  camera.intrinsics.validate_intrinsics();

  std::vector<float> map(2 * static_cast<std::size_t>(width) * height);

  for (uint32_t v = 0; v < height; ++v) {
    for (uint32_t u = 0; u < width; ++u) {
      const std::size_t entry =
          2 * (static_cast<std::size_t>(v) * width + u);

      const std::optional<Eigen::Vector2d> source_pixel =
          rectified_to_source_pixel(camera.intrinsics, camera.distortion,
                                    rectification, static_cast<double>(u),
                                    static_cast<double>(v));

      if (!source_pixel) {
        map[entry] = -1.0f;
        map[entry + 1] = -1.0f;
        continue;
      }

      // +0.5 to map integers from pixel centre to pixel boundary
      map[entry] = static_cast<float>(source_pixel->x() + 0.5);
      map[entry + 1] = static_cast<float>(source_pixel->y() + 0.5);
    }
  }
  return map;
}

namespace {

Rectification resize_projection(const Rectification &rectification,
                               ImageSize rectified_size, ImageSize target_size,
                               ResizeFit fit) {
  const CameraIntrinsics full = rectification.rectified_intrinsics();
  const CameraIntrinsics scaled_k =
      resize_intrinsics(full, rectified_size, target_size, fit);

  Rectification scaled = rectification;
  scaled.projection(0, 0) = scaled_k.fx;
  scaled.projection(1, 1) = scaled_k.fy;
  scaled.projection(0, 2) = scaled_k.cx;
  scaled.projection(1, 2) = scaled_k.cy;

  // P(0,3) is -fx * Tx
  scaled.projection(0, 3) =
      rectification.projection(0, 3) * resize_scale(rectified_size, target_size, fit);

  return scaled;
}

} // namespace

Rectification resize_rectification(const Rectification &rectification, ImageSize rectified_size,
                    ImageSize target_size, ResizeFit fit) {
  return resize_projection(rectification, rectified_size, target_size, fit);
}

StereoRectification resize_rectification(const StereoRectification &stereo_rect,
                                         ImageSize target_size, ResizeFit fit) {
  const ImageSize rectified = stereo_rect.size;

  StereoRectification scaled = stereo_rect;
  scaled.cameras[0] =
      resize_projection(stereo_rect.cameras[0], rectified, target_size, fit);
  scaled.cameras[1] =
      resize_projection(stereo_rect.cameras[1], rectified, target_size, fit);
  scaled.size = target_size;

  const CameraIntrinsics scaled_k = scaled.rectified_intrinsics();
  scaled.disparity_to_depth(0, 3) = -scaled_k.cx;
  scaled.disparity_to_depth(1, 3) = -scaled_k.cy;
  scaled.disparity_to_depth(2, 3) = scaled_k.fx;
  scaled.disparity_to_depth(3, 3) *= resize_scale(rectified, target_size, fit);

  scaled.validate();
  return scaled;
}

} // namespace perception::utils
