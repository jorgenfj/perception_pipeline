#ifndef PERCEPTION_GEOMETRY_STEREO_HPP
#define PERCEPTION_GEOMETRY_STEREO_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

#include <eigen3/Eigen/Dense>

#include <perception/geometry/camera_model.hpp>

namespace perception::geometry {

/**
 * @brief One camera's half of a stereo rectification
 *
 */
struct Rectification {
  /** R1 or R2: rotation taking this camera into the rectified frame.
   */
  Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};

  /** P1 or P2: the rectified projection, 3x4.
   *
   * The upper-left 3x3 block of the projection matrix
   * contains the common rectified intrinsic matrix.
   *
   */
  Eigen::Matrix<double, 3, 4> projection{Eigen::Matrix<double, 3, 4>::Zero()};

  /**
   * @brief The rectified camera's intrinsics — the left 3x3 of P.
   *
   * This is *not* the same as the camera's own K. Rectification generally
   * changes the focal length and principal point, and the rectified camera
   * has no distortion.
   */
  CameraIntrinsics rectified_intrinsics() const {
    return CameraIntrinsics{.fx = projection(0, 0),
                            .fy = projection(1, 1),
                            .cx = projection(0, 2),
                            .cy = projection(1, 2)};
  }

  /**
   * @brief P(0,3), which is -fx * baseline for the second camera and 0 for
   * the first. This is where the baseline enters the disparity maths.
   */
  double baseline_term() const { return projection(0, 3); }

  /**
   * @brief The stereo baseline this projection implies, in metres.
   *
   * P(0,3) is -fx * Tx and P(0,0) is fx, so the baseline is -P(0,3) / P(0,0).
   * Positive for the second camera of a left-right pair, and 0 for the first,
   * which carries no offset.
   *
   */
  double baseline() const { return -projection(0, 3) / projection(0, 0); }

  /**
   * @brief Build from the row-major arrays the calibration YAML stores.
   * @param r Row-major 3x3 rotation (R1 or R2).
   * @param p Row-major 3x4 projection (P1 or P2).
   * @return Rectification
   */
  static Rectification from_row_major(const std::array<double, 9> &r,
                                      const std::array<double, 12> &p) {
    Rectification rect;
    rect.rotation << r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8];
    rect.projection << p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
        p[9], p[10], p[11];

    const Eigen::Matrix3d &R = rect.rotation;
    if (!(R.transpose() * R).isApprox(Eigen::Matrix3d::Identity(), 1e-9) ||
        R.determinant() < 0.0) {
      throw std::runtime_error("Rectification: R is not a proper rotation");
    }

    CameraIntrinsics::validate_intrinsics(rect.projection.leftCols<3>());

    return rect;
  }
};

/**
 * @brief The rigid transform between the two cameras of a stereo pair, as
 *        cv::stereoCalibrate reports it.
 */
struct StereoExtrinsics {
  /** The second camera's pose in the first camera's frame. */
  Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};

  /** Translation in metres, second camera in the first's frame. */
  Eigen::Vector3d translation_m{Eigen::Vector3d::Zero()};

  /**
   * @brief |translation_m|, stated rather than derived.
   */
  double baseline_m{0.0};

  /** How far |translation_m| may sit from baseline_m before they disagree. */
  static constexpr double kBaselineTolerance_m = 1e-4;

  /**
   * @brief Check that this is a usable pair: a proper rotation, a positive
   * baseline, and a baseline_m that matches |translation_m|.
   *
   * @throws std::runtime_error naming the offending quantity.
   */
  void validate() const;

  /**
   * @brief Build from the row-major arrays a calibration file stores.
   * @param r Row-major 3x3 rotation, second camera in the first's frame.
   * @param t Translation in metres.
   * @param baseline The file's own statement of the baseline, in metres.
   * @return StereoExtrinsics, validated.
   *
   * @throws std::runtime_error if validate() rejects the result.
   */
  static StereoExtrinsics from_row_major(const std::array<double, 9> &r,
                                         const std::array<double, 3> &t,
                                         double baseline);
};

/**
 * @brief Both halves of one cv::stereoRectify run, and the Q that goes with
 *        them.
 *
 * Indexed to match the cameras it rectifies, so cameras[0] is the reference camera,
 * the one stereoRectify leaves at the origin of the rectified frame, and the
 * one that therefore carries no baseline offset in its projection.
 */
struct StereoRectification {
  Rectification cameras[2];

  /** Q: [X Y Z W]^T = Q * [u v disparity 1]^T. */
  Eigen::Matrix4d disparity_to_depth{Eigen::Matrix4d::Zero()};

  /**
   * @brief The rectified image size these projections were computed for
   */
  ImageSize size;

  /**
   * @brief The rectified intrinsics both cameras share.
   */
  CameraIntrinsics rectified_intrinsics() const {
    return cameras[0].rectified_intrinsics();
  }

  /**
   * @brief The rectified focal length in pixels, for consumers computing
   * depth = fx * baseline / disparity.
   * @return fx of the common rectified intrinsics.
   */
  double rectified_fx() const { return rectified_intrinsics().fx; }

  /**
   * @brief The baseline the rectified projections imply, in metres.
   *
   * The whole baseline lives in cameras[1]: P2(0,3) is -fx * Tx and the reference
   * camera carries no offset.
   *
   * @return |-P2(0,3) / P2(0,0)|.
   */
  double baseline_m() const { return std::abs(cameras[1].baseline()); }

  /** How far the two cameras' shared intrinsics may sit apart, in pixels. */
  static constexpr double kIntrinsicsTolerance_px = 1e-6;

  /**
   * @brief Check that the two halves belong together, and that `size` was set
   * and holds the rectified principal point.
   *
   * @throws std::runtime_error naming what disagrees.
   */
  void validate() const;

  /**
   * @brief Check the rectified baseline against the extrinsics' own.
   *
   * @param extrinsics The pair this rectification was computed for.
   * @param tolerance_m Allowed disagreement, in metres.
   *
   * @throws std::runtime_error if they disagree by more than tolerance_m.
   */
  void validate_against(const StereoExtrinsics &extrinsics,
                        double tolerance_m = 1e-3) const;
};

/**
 * @brief Whether both cameras' rectifying rotations land in the same rectified
 *        orientation.
 *
 * @param rectification The pair's R1 and R2.
 * @param extrinsics The R the pair was rectified from.
 * @param tolerance Passed to Eigen's isApprox.
 * @return true if R2 * R matches R1.
 */
bool rectifying_rotations_agree(const StereoRectification &rectification,
                                const StereoExtrinsics &extrinsics,
                                double tolerance = 1e-9);

/**
 * @brief A calibrated stereo pair: two cameras, the transform between them,
 *        and the rectification derived from both.
 */
struct StereoCalibration {
  /**
   * @brief The size of the source images the two cameras were calibrated at.
   */
  ImageSize size;

  /** The two unrectified cameras. cameras[0] is the reference camera. */
  PinholeCameraModel cameras[2];

  StereoExtrinsics extrinsics;
  StereoRectification rectification;

  /**
   * @brief Every member's own checks, plus the ones that span them.
   *
   * For a calibration assembled in memory. A loader may prefer the members'
   * validate() as it fills them, so a rejection can name the key it came from.
   *
   * @throws std::runtime_error naming what is wrong.
   */
  void validate() const;

  /** One line for a startup log. */
  std::string summary() const;
};

/**
 * @brief For one pixel of the rectified output, get the pixel in the
 * *source* image that it samples.
 *
 * This is the body of cv::initUndistortRectifyMap: invert the rectifying
 * rotation and the rectified projection to get the ray this output pixel
 * looks along, re-apply the lens distortion, then project through the
 * camera's own K.
 *
 * Note that P(0,3) plays no part. The fourth column of P translates the
 * rectified camera along its own x axis, which moves where the *other* camera
 * sees a point, not where this camera samples.
 *
 * @param intrinsics This camera's unrectified K.
 * @param distortion This camera's lens distortion.
 * @param rectification This camera's R and P from stereoRectify.
 * @param u Rectified pixel column index.
 * @param v Rectified pixel row index.
 * @return Source pixel coordinate (u, v) in pixel-index convention, or
 *         std::nullopt if this rectified pixel looks behind the camera and
 *         no source pixel exists.
 */
std::optional<Eigen::Vector2d> rectified_to_source_pixel(
    const CameraIntrinsics &intrinsics, const CameraDistortionModel &distortion,
    const Rectification &rectification, double u, double v);

} // namespace perception::geometry

#endif // PERCEPTION_GEOMETRY_STEREO_HPP
