#include <gtest/gtest.h>

#include <perception/geometry/camera_model.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using perception::geometry::CameraDistortionModel;
using perception::geometry::CameraIntrinsics;
using perception::geometry::Rectification;
using perception::geometry::rectified_to_source_pixel;

namespace {

constexpr double kEps = 1e-10;

/**
 * Absolute comparison of two 2-vectors.
 *
 * Eigen's isApprox() is *relative*, so it can never pass against the zero
 * vector -- and pixel (0, 0) and the distortion centre are both exactly that.
 * These are pixels and normalized image coordinates, quantities with a
 * meaningful absolute scale, so an absolute tolerance is the right test.
 */
::testing::AssertionResult VectorsNear(const Eigen::Vector2d& actual,
                                       const Eigen::Vector2d& expected,
                                       double tolerance) {
    const double error = (actual - expected).norm();
    if (error <= tolerance) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << "(" << actual.x() << ", " << actual.y() << ") vs expected ("
           << expected.x() << ", " << expected.y() << "), error " << error
           << " > " << tolerance;
}

/** The shipped left eye, from app/config/stereo_calibration.yaml. */
CameraIntrinsics shipped_left_intrinsics() {
    return CameraIntrinsics{.fx = 2329.7529233353694,
                            .fy = 2329.5740433167261,
                            .cx = 754.5022389652534,
                            .cy = 560.6093817146035};
}

CameraDistortionModel shipped_left_distortion() {
    return CameraDistortionModel{.k1 = -0.24410080138920398,
                                 .k2 = 0.26696399429948375,
                                 .p1 = 8.827990083798891e-05,
                                 .p2 = 2.4006501080748617e-05,
                                 .k3 = -0.15345901438237441};
}

Rectification shipped_left_rectification() {
    return Rectification::from_row_major(
        {0.9953566462359332, 0.09624063867688702, 0.001698899944114407,
         -0.09624479980195409, 0.9953543863342791, 0.0025659532716845275,
         -0.0014440585296353384, -0.0027175489279068757, 0.9999952648001825},
        {2619.5697059803665, 0.0, 745.2493591308594, 0.0, 0.0,
         2619.5697059803665, 558.5332374572754, 0.0, 0.0, 0.0, 1.0, 0.0});
}

}  // namespace

class CameraIntrinsicsTests : public ::testing::Test {
   public:
    CameraIntrinsicsTests() = default;
    void SetUp() override {}
};

TEST_F(CameraIntrinsicsTests, test_default_initialization) {
    CameraIntrinsics intrinsics;
    EXPECT_EQ(intrinsics.fx, 0.0);
    EXPECT_EQ(intrinsics.fy, 0.0);
    EXPECT_EQ(intrinsics.cx, 0.0);
    EXPECT_EQ(intrinsics.cy, 0.0);
    EXPECT_EQ(intrinsics.skew, 0.0);
}

TEST_F(CameraIntrinsicsTests, test_k_inverse_is_the_inverse_of_k) {
    // Without skew, which is what every standard calibration produces.
    const CameraIntrinsics intrinsics = shipped_left_intrinsics();
    const Eigen::Matrix3d product = intrinsics.K() * intrinsics.K_inv();
    EXPECT_TRUE(product.isApprox(Eigen::Matrix3d::Identity(), kEps));
}

TEST_F(CameraIntrinsicsTests, test_k_inverse_handles_skew) {
    // The skew branch of K_inv() is separate arithmetic and is never exercised
    // by this rig's calibration, so it gets its own case.
    CameraIntrinsics intrinsics = shipped_left_intrinsics();
    intrinsics.skew = 3.7;
    const Eigen::Matrix3d product = intrinsics.K() * intrinsics.K_inv();
    EXPECT_TRUE(product.isApprox(Eigen::Matrix3d::Identity(), kEps));
}

TEST_F(CameraIntrinsicsTests, test_project_backproject_round_trip) {
    const CameraIntrinsics intrinsics = shipped_left_intrinsics();

    for (const Eigen::Vector2d& pixel :
         {Eigen::Vector2d{0.0, 0.0}, Eigen::Vector2d{754.5, 560.6},
          Eigen::Vector2d{1439.0, 1079.0}, Eigen::Vector2d{12.0, 900.0}}) {
        const Eigen::Vector3d point = intrinsics.backproject_point(pixel, 4.2);
        const Eigen::Vector2d reprojected = intrinsics.project_point(point);
        EXPECT_TRUE(VectorsNear(reprojected, pixel, kEps));
    }
}

TEST_F(CameraIntrinsicsTests, test_round_trip_survives_skew) {
    CameraIntrinsics intrinsics = shipped_left_intrinsics();
    intrinsics.skew = 3.7;

    const Eigen::Vector2d pixel{300.0, 900.0};
    const Eigen::Vector3d point = intrinsics.backproject_point(pixel, 2.0);
    EXPECT_TRUE(VectorsNear(intrinsics.project_point(point), pixel, kEps));
}

TEST_F(CameraIntrinsicsTests, test_backproject_ray_has_unit_z) {
    const CameraIntrinsics intrinsics = shipped_left_intrinsics();
    const Eigen::Vector3d ray =
        intrinsics.backproject_ray(Eigen::Vector2d{100.0, 200.0});
    EXPECT_DOUBLE_EQ(ray.z(), 1.0);
}

TEST_F(CameraIntrinsicsTests, test_principal_point_backprojects_along_axis) {
    const CameraIntrinsics intrinsics = shipped_left_intrinsics();
    const Eigen::Vector3d ray = intrinsics.backproject_ray(
        Eigen::Vector2d{intrinsics.cx, intrinsics.cy});
    EXPECT_NEAR(ray.x(), 0.0, kEps);
    EXPECT_NEAR(ray.y(), 0.0, kEps);
}

TEST_F(CameraIntrinsicsTests, test_row_major_round_trip) {
    const CameraIntrinsics intrinsics = shipped_left_intrinsics();
    const CameraIntrinsics restored =
        CameraIntrinsics::from_row_major(intrinsics.to_row_major());

    EXPECT_DOUBLE_EQ(restored.fx, intrinsics.fx);
    EXPECT_DOUBLE_EQ(restored.fy, intrinsics.fy);
    EXPECT_DOUBLE_EQ(restored.cx, intrinsics.cx);
    EXPECT_DOUBLE_EQ(restored.cy, intrinsics.cy);
    EXPECT_DOUBLE_EQ(restored.skew, intrinsics.skew);
}

TEST_F(CameraIntrinsicsTests, test_from_row_major_reads_opencv_layout) {
    // Row-major [fx s cx; 0 fy cy; 0 0 1] -- a transpose here would swap cx
    // into the skew slot and go unnoticed on a symmetric image.
    const CameraIntrinsics intrinsics = CameraIntrinsics::from_row_major(
        {100.0, 0.5, 320.0, 0.0, 200.0, 240.0, 0.0, 0.0, 1.0});

    EXPECT_DOUBLE_EQ(intrinsics.fx, 100.0);
    EXPECT_DOUBLE_EQ(intrinsics.skew, 0.5);
    EXPECT_DOUBLE_EQ(intrinsics.cx, 320.0);
    EXPECT_DOUBLE_EQ(intrinsics.fy, 200.0);
    EXPECT_DOUBLE_EQ(intrinsics.cy, 240.0);
}

TEST_F(CameraIntrinsicsTests, test_rejects_non_positive_focals) {
    CameraIntrinsics intrinsics = shipped_left_intrinsics();
    intrinsics.fx = 0.0;
    EXPECT_THROW(intrinsics.validate_focals(), std::runtime_error);

    // NOTE: K(), K_inv(), project_point() and backproject_ray() no longer call
    // validate_focals() -- only rectified_to_source_pixel() does. So K_inv()
    // with fx = 0 returns inf rather than throwing. Asserted here as the
    // current contract; see the note in the review if that was not intended.
    EXPECT_NO_THROW(intrinsics.K());
    EXPECT_FALSE(std::isfinite(intrinsics.K_inv()(0, 0)));

    intrinsics.fx = 100.0;
    intrinsics.fy = -1.0;
    EXPECT_THROW(intrinsics.validate_focals(), std::runtime_error);
}

TEST_F(CameraIntrinsicsTests, test_rejects_projection_behind_the_camera) {
    const CameraIntrinsics intrinsics = shipped_left_intrinsics();
    EXPECT_THROW(intrinsics.project_point(Eigen::Vector3d{1.0, 1.0, 0.0}),
                 std::runtime_error);
    EXPECT_THROW(intrinsics.project_point(Eigen::Vector3d{1.0, 1.0, -2.0}),
                 std::runtime_error);
}

TEST_F(CameraIntrinsicsTests, test_rejects_non_positive_depth) {
    const CameraIntrinsics intrinsics = shipped_left_intrinsics();
    const Eigen::Vector2d pixel{100.0, 100.0};
    EXPECT_THROW(intrinsics.backproject_point(pixel, 0.0), std::runtime_error);
    EXPECT_THROW(intrinsics.backproject_point(pixel, -1.0),
                 std::runtime_error);
}

class CameraDistortionModelTests : public ::testing::Test {
   public:
    CameraDistortionModelTests() = default;
    void SetUp() override {}
};

TEST_F(CameraDistortionModelTests, test_zero_model_is_identity) {
    const CameraDistortionModel distortion;
    const Eigen::Vector2d point{0.13, -0.07};
    EXPECT_TRUE(VectorsNear(distortion.distort_normalized(point), point, kEps));
    EXPECT_TRUE(
        VectorsNear(distortion.undistort_normalized(point), point, kEps));
}

TEST_F(CameraDistortionModelTests, test_undistort_inverts_distort) {
    const CameraDistortionModel distortion = shipped_left_distortion();

    // Normalized coordinates spanning the calibrated image: the corner of a
    // 1440x1080 frame at fx ~2330 sits around (0.31, 0.22).
    for (const Eigen::Vector2d& point :
         {Eigen::Vector2d{0.0, 0.0}, Eigen::Vector2d{0.10, 0.05},
          Eigen::Vector2d{-0.31, 0.22}, Eigen::Vector2d{0.31, -0.22}}) {
        const Eigen::Vector2d distorted =
            distortion.distort_normalized(point);
        const Eigen::Vector2d recovered =
            distortion.undistort_normalized(distorted);
        EXPECT_TRUE(VectorsNear(recovered, point, kEps));
    }
}

TEST_F(CameraDistortionModelTests, test_distortion_actually_moves_a_point) {
    // Guards against an undistort that "passes" by both directions being the
    // identity -- at k1 = -0.244 the frame corner must move appreciably.
    const CameraDistortionModel distortion = shipped_left_distortion();
    const Eigen::Vector2d corner{0.31, 0.22};
    const Eigen::Vector2d distorted = distortion.distort_normalized(corner);
    EXPECT_GT((distorted - corner).norm(), 1e-3);
}

TEST_F(CameraDistortionModelTests, test_from_coefficients_requires_exactly_five) {
    const CameraDistortionModel distortion =
        CameraDistortionModel::from_coefficients({1.0, 2.0, 3.0, 4.0, 5.0});

    EXPECT_DOUBLE_EQ(distortion.k1, 1.0);
    EXPECT_DOUBLE_EQ(distortion.k2, 2.0);
    EXPECT_DOUBLE_EQ(distortion.p1, 3.0);
    EXPECT_DOUBLE_EQ(distortion.p2, 4.0);
    EXPECT_DOUBLE_EQ(distortion.k3, 5.0);

    // A short vector is a truncated model, not a shorter one: refused rather
    // than zero-padded, so a file missing k3 is a parse error and not a
    // silently different lens.
    EXPECT_THROW(CameraDistortionModel::from_coefficients({1.0, 2.0, 3.0, 4.0}),
                 std::runtime_error);
    EXPECT_THROW(CameraDistortionModel::from_coefficients({}),
                 std::runtime_error);
}

TEST_F(CameraDistortionModelTests, test_from_coefficients_rejects_rational) {
    // Eight coefficients is OpenCV's rational model. Truncating it would give
    // a model that is wrong everywhere and never looks wrong.
    const std::vector<double> rational(8, 0.1);
    EXPECT_THROW(CameraDistortionModel::from_coefficients(rational),
                 std::runtime_error);
}

TEST_F(CameraDistortionModelTests, test_coefficients_round_trip) {
    const CameraDistortionModel distortion = shipped_left_distortion();
    const CameraDistortionModel restored =
        CameraDistortionModel::from_coefficients(distortion.to_coefficients());

    EXPECT_DOUBLE_EQ(restored.k1, distortion.k1);
    EXPECT_DOUBLE_EQ(restored.k2, distortion.k2);
    EXPECT_DOUBLE_EQ(restored.p1, distortion.p1);
    EXPECT_DOUBLE_EQ(restored.p2, distortion.p2);
    EXPECT_DOUBLE_EQ(restored.k3, distortion.k3);
}

// Convergence of undistort_normalized() over the whole calibrated frame.
//
// The fixed-point iteration diverges outside a finite radius, so "does it
// converge" is a property of the coefficients AND the domain, not of the code.
// This sweeps every point the calibrated image can produce, which is what makes
// the answer binding for this rig rather than for the handful of points the
// other tests happen to pick.
class UndistortConvergenceTests : public ::testing::Test {
   public:
    UndistortConvergenceTests() = default;
    void SetUp() override {}
};

TEST_F(UndistortConvergenceTests, test_converges_across_the_whole_frame) {
    const CameraIntrinsics intrinsics = shipped_left_intrinsics();
    const CameraDistortionModel distortion = shipped_left_distortion();

    // A 4-pixel grid over 1440x1080: 360 x 270 = 97200 points, every one of
    // them a coordinate the map builder could actually be handed.
    constexpr int kStep = 4;
    double worst = 0.0;
    double worst_radius = 0.0;

    for (int py = 0; py < 1080; py += kStep) {
        for (int px = 0; px < 1440; px += kStep) {
            const Eigen::Vector2d normalized{
                (px - intrinsics.cx) / intrinsics.fx,
                (py - intrinsics.cy) / intrinsics.fy};
            worst_radius = std::max(worst_radius, normalized.norm());

            // Feed the DISTORTED point, which is what a caller measuring in a
            // real image would have.
            const Eigen::Vector2d distorted =
                distortion.distort_normalized(normalized);

            Eigen::Vector2d recovered;
            ASSERT_NO_THROW(recovered =
                                distortion.undistort_normalized(distorted))
                << "pixel (" << px << ", " << py << ")";

            worst = std::max(worst, (recovered - normalized).norm());
        }
    }

    EXPECT_LT(worst, 1e-12) << "worst round trip " << worst;
    // Sanity that the sweep really covered the frame corners.
    EXPECT_GT(worst_radius, 0.40);
}

TEST_F(UndistortConvergenceTests, test_ten_iterations_suffice_over_the_frame) {
    // The default budget is 40. Establishing that the frame needs ~10 is what
    // says the margin is real rather than the default merely being generous.
    const CameraIntrinsics intrinsics = shipped_left_intrinsics();
    const CameraDistortionModel distortion = shipped_left_distortion();

    for (int py = 0; py < 1080; py += 16) {
        for (int px = 0; px < 1440; px += 16) {
            const Eigen::Vector2d normalized{
                (px - intrinsics.cx) / intrinsics.fx,
                (py - intrinsics.cy) / intrinsics.fy};
            const Eigen::Vector2d distorted =
                distortion.distort_normalized(normalized);
            EXPECT_NO_THROW(distortion.undistort_normalized(distorted, 10));
        }
    }
}

TEST_F(UndistortConvergenceTests, test_throws_outside_the_invertible_radius) {
    // r = 1.0 is well past this model's limit of about 0.891. An unchecked
    // iteration returns roughly 1e34 here; this must refuse instead.
    const CameraDistortionModel distortion = shipped_left_distortion();

    EXPECT_THROW(distortion.undistort_normalized(Eigen::Vector2d{1.0, 0.0}),
                 std::runtime_error);
    EXPECT_THROW(distortion.undistort_normalized(Eigen::Vector2d{0.0, -1.0}),
                 std::runtime_error);
    EXPECT_THROW(distortion.undistort_normalized(Eigen::Vector2d{2.0, 2.0}),
                 std::runtime_error);
}

TEST_F(UndistortConvergenceTests, test_the_frame_sits_inside_the_limit) {
    // Where the limit actually is, bisected, so a future calibration with
    // stronger distortion fails here loudly instead of somewhere downstream.
    const CameraDistortionModel distortion = shipped_left_distortion();

    const auto converges = [&distortion](double radius) {
        for (int i = 0; i < 64; ++i) {
            const double angle = 2.0 * M_PI * i / 64.0;
            try {
                distortion.undistort_normalized(Eigen::Vector2d{
                    radius * std::cos(angle), radius * std::sin(angle)});
            } catch (const std::runtime_error&) {
                return false;
            }
        }
        return true;
    };

    double lo = 0.5;
    double hi = 1.2;
    for (int i = 0; i < 40; ++i) {
        const double mid = 0.5 * (lo + hi);
        (converges(mid) ? lo : hi) = mid;
    }

    // The calibrated frame reaches r = 0.4035. Demand real headroom, not a
    // hairline pass.
    EXPECT_GT(lo, 0.404 * 1.5) << "invertible radius " << lo
                               << " leaves too little margin over the frame";
    std::printf("  [ INFO ] invertible radius %.5f, frame corner 0.40350, "
                "margin %.2fx\n",
                lo, lo / 0.4035);
}