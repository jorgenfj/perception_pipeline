#include "transforms/rectify_map.hpp"

#include <cstddef>
#include <optional>
#include <stdexcept>

#include <perception/geometry/camera_model.hpp>

namespace perception {

// Host-only, and in a .cpp rather than the .cu next to it on purpose: this is
// pure camera maths with no device code in it, and keeping it out of the .cu
// keeps nvcc from having to compile Eigen.

std::vector<float2> build_rectify_map(const CameraCalibration& camera, uint32_t width,
                                      uint32_t height, RectifyMapCoords coords) {
  if (width == 0 || height == 0) {
    throw std::runtime_error("build_rectify_map: zero-sized image");
  }

  // Once, here, rather than per pixel inside rectified_to_source_pixel: the
  // intrinsics cannot change during the loop, and an fx of 0 would otherwise
  // put the principal point in every entry of the map without complaint.
  camera.intrinsics.validate_intrinsics();

  // Texture coordinates put texel centres on the half-integers, so pixel index
  // i is sampled at i + 0.5. The map holds texture coordinates, not indices,
  // so the half-pixel is folded in here rather than in the kernel.
  const double u_scale = (coords == RectifyMapCoords::Normalized) ? 1.0 / width : 1.0;
  const double v_scale = (coords == RectifyMapCoords::Normalized) ? 1.0 / height : 1.0;

  std::vector<float2> map(static_cast<std::size_t>(width) * height);

  for (uint32_t v = 0; v < height; ++v) {
    for (uint32_t u = 0; u < width; ++u) {
      const std::optional<Eigen::Vector2d> source =
          geometry::rectified_to_source_pixel(
              camera.intrinsics, camera.distortion, camera.rectification,
              static_cast<double>(u), static_cast<double>(v));

      if (!source) {
        map[static_cast<std::size_t>(v) * width + u] = make_float2(-1.0f, -1.0f);
        continue;
      }

      map[static_cast<std::size_t>(v) * width + u] =
          make_float2(static_cast<float>((source->x() + 0.5) * u_scale),
                      static_cast<float>((source->y() + 0.5) * v_scale));
    }
  }
  return map;
}

std::size_t count_out_of_frame(const std::vector<float2>& map, uint32_t width, uint32_t height,
                               RectifyMapCoords coords) {
  const float u_limit = (coords == RectifyMapCoords::Normalized) ? 1.0f : static_cast<float>(width);
  const float v_limit =
      (coords == RectifyMapCoords::Normalized) ? 1.0f : static_cast<float>(height);

  std::size_t count = 0;
  for (const float2& m : map) {
    if (m.x < 0.0f || m.x >= u_limit || m.y < 0.0f || m.y >= v_limit) ++count;
  }
  return count;
}

}  // namespace perception
