#ifndef PERCEPTION_GEOMETRY_CAMERA_MODEL_HPP
#define PERCEPTION_GEOMETRY_CAMERA_MODEL_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <eigen3/Eigen/Dense>
#include <format>
#include <stdexcept>
#include <vector>

namespace perception::geometry {

/**
 * @brief The pixel extent of an image.
 */
struct ImageSize {
  uint32_t width{0};
  uint32_t height{0};

  /** True if either extent is zero, i.e. the size was never set. */
  bool empty() const { return width == 0 || height == 0; }

  bool operator==(const ImageSize &) const = default;
};

/**
 * @brief Pinhole camera intrinsics.
 *
 * Pixel coordinate convention: pixel (i, j) has its *center* at coordinate
 * (i, j), so the image spans [-0.5, W-0.5] x [-0.5, H-0.5]. This is
 * OpenCV's convention. Note it differs by half a pixel from the texture-space
 * convention used by OpenGL and most GPU rasterizers.
 *
 * K = [fx, 0.0, cx]
 *     [0.0, fy, cy]
 *     [0.0 0.0, 1.0]
 *
 * Non-zero skew is not supported.
 */
struct CameraIntrinsics {
  double fx{};
  double fy{};
  double cx{};
  double cy{};

  /**
   * @brief Check that a 3x3 matrix is a camera matrix this model can
   * represent: upper triangular with a [0 0 1] bottom row, zero skew, finite
   * principal point and strictly positive focals.
   *
   * @param k Candidate camera matrix.
   *
   * @throws std::runtime_error if any of the above does not hold.
   */
  /**
   * Absolute tolerance for K's structural entries: the zeros below the
   * diagonal, the zero skew, and the 1 in the corner.
   *
   */
  static constexpr double kStructureTolerance = 1e-9;

  static void validate_intrinsics(const Eigen::Matrix3d &k) {
    const auto is_zero = [](double value) {
      return std::abs(value) <= kStructureTolerance;
    };

    if (!is_zero(k(1, 0)) || !is_zero(k(2, 0)) || !is_zero(k(2, 1)) ||
        !(std::abs(k(2, 2) - 1.0) <= kStructureTolerance)) {
      throw std::runtime_error(std::format(
          "Invalid Camera Matrix K. K must be upper triangular with bottom "
          "row [0 0 1]:\n"
          "[{}, {}, {}]\n[{}, {}, {}]\n[{}, {}, {}]",
          k(0, 0), k(0, 1), k(0, 2), k(1, 0), k(1, 1), k(1, 2), k(2, 0),
          k(2, 1), k(2, 2)));
    }
    if (!is_zero(k(0, 1))) {
      throw std::runtime_error(std::format(
          "Invalid Camera Matrix K. Non-zero skew ({}) is not supported.",
          k(0, 1)));
    }
    if (!(k(0, 0) > 0.0) || !(k(1, 1) > 0.0) || !std::isfinite(k(0, 0)) ||
        !std::isfinite(k(1, 1))) {
      throw std::runtime_error(std::format(
          "Invalid Camera Matrix K. Focals fx and fy must be positive and "
          "finite:\nfx = {}, fy = {}",
          k(0, 0), k(1, 1)));
    }
    if (!std::isfinite(k(0, 2)) || !std::isfinite(k(1, 2))) {
      throw std::runtime_error(std::format(
          "Invalid Camera Matrix K. Principal point must be finite:\n"
          "cx = {}, cy = {}",
          k(0, 2), k(1, 2)));
    }
  }

  /**
   * @brief Validate this camera's own K().
   *
   * The structural checks hold by construction here -- K() builds the zeros
   * and the 1 -- so in practice this validates the focals and the principal
   * point. It shares one implementation, and one error text, with the check
   * applied to a K read off disk.
   *
   * @throws std::runtime_error if the intrinsics are not usable.
   */
  void validate_intrinsics() const { validate_intrinsics(K()); }

  /**
   * @brief Get the camera intrinsic matrix K
   * according to Eq. (2.57) in Szeliski, 2022.
   * @return Eigen::Matrix3d K
   */
  Eigen::Matrix3d K() const {
    Eigen::Matrix3d k = Eigen::Matrix3d::Identity();
    k(0, 0) = fx;
    k(0, 2) = cx;
    k(1, 1) = fy;
    k(1, 2) = cy;

    return k;
  }

  /**
   * @brief Get the inverse camera intrinsic matrix K
   * @return Eigen::Matrix3d K_inv
   */
  Eigen::Matrix3d K_inv() const {
    Eigen::Matrix3d k_inv = Eigen::Matrix3d::Identity();
    k_inv(0, 0) = 1.0 / fx;
    k_inv(1, 1) = 1.0 / fy;
    k_inv(1, 2) = -cy / fy;
    k_inv(0, 2) = -cx / fx; // skew = 0

    return k_inv;
  }

  /**
   * @brief Map a pixel to normalized image coordinates, the point where its
   * ray crosses the z = 1 plane.
   *
   * This is K_inv() applied to the pixel in homogeneous form, written out
   * rather than built as a matrix product.
   *
   * @param pixel Eigen::Vector2d representing camera pixel coordinates.
   * @return 2D point in normalized image coordinates.
   */
  Eigen::Vector2d pixel_to_normalized(const Eigen::Vector2d &pixel) const {
    return {(pixel.x() - cx) / fx, (pixel.y() - cy) / fy};
  }

  /**
   * @brief Map normalized image coordinates back to pixel coordinates.
   *
   * The the mapping from normalized coords to pixel coords using K,
   * written out rather than built as a matrix product.
   *
   * @param normalized 2D point in normalized image coordinates.
   * @return 2D pixel coordinates (u, v).
   */
  Eigen::Vector2d normalized_to_pixel(const Eigen::Vector2d &normalized) const {
    return {fx * normalized.x() + cx, fy * normalized.y() + cy};
  }

  /**
   * @brief Project a 3D point expressed in the camera coordinate frame
   *        onto the image sensor, producing pixel coordinates.
   *
   * This function implements the pinhole camera model by:
   *  1) applying perspective division (Eq. 2.50 in Szeliski),
   *  2) mapping normalized image coordinates to pixel coordinates
   *     using the camera intrinsics (Eq. 2.54 with K defined in Eq. 2.57).
   *
   * @param point 3D point in camera coordinates (Xc, Yc, Zc), with Zc > 0.
   * @return 2D pixel coordinates (u, v).
   *
   * @throws std::runtime_error if Zc <= 0.
   */
  Eigen::Vector2d project_point(const Eigen::Vector3d &point) const {
    if (point.z() <= 0.0) {
      throw std::runtime_error(
          "Projection of point failed. Can't project with z <= 0.");
    }
    return normalized_to_pixel(point.hnormalized());
  }

  /**
   * @brief Backproject a pixel using the inverse intrinsics matrix to produce
   * a ray through the pixel. The ray is in normalized image coordinates with
   * z = 1.0 and is not necessarily normalized.
   * @param pixel Eigen::Vector2d representing camera pixel coordinates.
   * @return ray in camera space (x, y, 1.0).
   */
  Eigen::Vector3d backproject_ray(const Eigen::Vector2d &pixel) const {
    return pixel_to_normalized(pixel).homogeneous();
  }

  /**
   * @brief Backproject a pixel using the inverse intrinsics matrix and a
   * depth value to compute the corresponding 3D point.
   * @param pixel Eigen::Vector2d representing camera pixel coordinates.
   * @param depth Depth for the corresponding pixel.
   * @return 3D point in camera space
   *
   * @throws std::runtime_error if depth <= 0.
   */
  Eigen::Vector3d backproject_point(const Eigen::Vector2d &pixel,
                                    double depth) const {
    if (depth <= 0.0) {
      throw std::runtime_error(
          "Backprojection failed. Depth must be positive.");
    }
    return depth * backproject_ray(pixel);
  }

  /**
   * @brief Build intrinsics from a row-major 3x3 K, the layout OpenCV dumps
   * and the calibration YAML stores.
   * @param k Row-major [fx s cx; 0 fy cy; 0 0 1].
   * @return CameraIntrinsics
   */
  static CameraIntrinsics from_row_major(const std::array<double, 9> &k) {
    Eigen::Matrix3d m;
    m << k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8];
    validate_intrinsics(m);

    return CameraIntrinsics{.fx = k[0], .fy = k[4], .cx = k[2], .cy = k[5]};
  }

  /**
   * @brief Inverse of from_row_major(), for writing a calibration back out.
   * @return Row-major 3x3 K.
   */
  std::array<double, 9> to_row_major() const {
    this->validate_intrinsics();
    return {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
  }
};

/**
 * @brief Brown-Conrady (OpenCV "plumb_bob") lens distortion, [k1 k2 p1 p2 k3].
 */
struct CameraDistortionModel {
  double k1{0.0};
  double k2{0.0};
  double p1{0.0};
  double p2{0.0};
  double k3{0.0};

  /**
   * @brief Distort a point in normalized image coordinates following the
   * Brown-Conrady forward distortion model. This matches the OpenCV
   * convention.
   * @param point Eigen::Vector2d representing and undistorted point in
   * normalized image coordinates.
   * @return 2D distorted point in normalized image coordinates.
   */
  Eigen::Vector2d distort_normalized(const Eigen::Vector2d &point) const {
    const double x = point.x();
    const double y = point.y();
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;

    const double r_dist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    const double t_dist_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double t_dist_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

    const double x_dist = r_dist * x + t_dist_x;
    const double y_dist = r_dist * y + t_dist_y;

    return {x_dist, y_dist};
  }

  /** Convergence target for undistort_normalized(), in normalized units. */
  static constexpr double kUndistortTolerance = 1e-9;

  /**
   * @brief Invert distort_normalized() by fixed-point iteration on the
   * forward model, as cv::undistortPoints does.
   *
   * There is no closed form: the forward model is a polynomial in the
   * undistorted radius, so the inverse is found by repeatedly solving for the
   * radial term and subtracting the tangential one.
   *
   * It does not converge everywhere, and the iterate can grow without bound
   *
   * @param point Distorted point in normalized image coordinates.
   * @param iterations Fixed-point steps.
   * @return Undistorted point in normalized image coordinates.
   *
   * @throws std::runtime_error if the iteration has not converged to
   *         kUndistortTolerance within `iterations` steps
   */
  Eigen::Vector2d undistort_normalized(const Eigen::Vector2d &point,
                                       int iterations = 40) const;

  /**
   * @brief Build the model from a coefficient vector in OpenCV's order.
   *
   * @param coefficients [k1 k2 p1 p2 k3], exactly five.
   * @return CameraDistortionModel
   *
   * @throws std::runtime_error unless exactly five coefficients are given.
   */
  static CameraDistortionModel
  from_coefficients(const std::vector<double> &coefficients) {
    if (coefficients.size() != 5) {
      throw std::runtime_error(std::format(
          "{} distortion coefficients; plumb_bob takes 5 "
          "[k1 k2 p1 p2 k3], and the rational/thin-prism/tilted terms are "
          "not implemented",
          coefficients.size()));
    }

    return CameraDistortionModel{.k1 = coefficients.at(0),
                                 .k2 = coefficients.at(1),
                                 .p1 = coefficients.at(2),
                                 .p2 = coefficients.at(3),
                                 .k3 = coefficients.at(4)};
  }

  /**
   * @brief Inverse of from_coefficients(), always five long.
   * @return [k1 k2 p1 p2 k3]
   */
  std::vector<double> to_coefficients() const { return {k1, k2, p1, p2, k3}; }
};

/**
 * @brief Pinhole Camera Model
 */
struct PinholeCameraModel {
  CameraIntrinsics intrinsics{};
  CameraDistortionModel distortion{};

  /**
   * @brief Project a 3D point expressed in the camera coordinate frame onto
   *        the image sensor, producing a pixel in the distorted (raw) image.
   *
   * The full forward model of a pinhole camera:
   *  1) perspective division (Eq. 2.50 in Szeliski),
   *  2) the Brown-Conrady forward distortion,
   *  3) normalized image coordinates to pixels through K (Eq. 2.54).
   *
   * @param point 3D point in camera coordinates (Xc, Yc, Zc), with Zc > 0.
   * @return 2D pixel coordinates (u, v) in the distorted image.
   *
   * @throws std::runtime_error if Zc <= 0.
   */
  Eigen::Vector2d project_point(const Eigen::Vector3d &point) const {
    if (point.z() <= 0.0) {
      throw std::runtime_error(
          "Projection of point failed. Can't project with z <= 0.");
    }
    return intrinsics.normalized_to_pixel(
        distortion.distort_normalized(point.hnormalized()));
  }

  /**
   * @brief Backproject a pixel of the distorted (raw) image to produce a ray
   * through it in the camera frame. The ray crosses z = 1.0 and is not
   * normalized.
   *
   * The inverse operation of project_point() defined up to scale
   * (returns direction vector).
   *
   * 1) Convert pixel to normalized image coordinates,
   * 2) Apply iterative undistortion (not guaranteed to converge),
   * 3) Convert to homogenous form by appending 1.0 as Z value.
   *
   * @param pixel Eigen::Vector2d representing distorted camera pixel
   * coordinates.
   * @return ray in camera space (x, y, 1.0).
   *
   * @throws std::runtime_error if the undistortion does not converge; see
   *         CameraDistortionModel::undistort_normalized().
   */
  Eigen::Vector3d backproject_pixel(const Eigen::Vector2d &pixel) const {
    return distortion
        .undistort_normalized(intrinsics.pixel_to_normalized(pixel))
        .homogeneous();
  }
};

} // namespace perception::geometry

#endif // PERCEPTION_GEOMETRY_CAMERA_MODEL_HPP
