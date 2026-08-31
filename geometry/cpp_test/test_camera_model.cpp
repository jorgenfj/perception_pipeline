#include <gtest/gtest.h>

#include <perception/geometry/camera_model.hpp>
#include <perception/geometry/stereo.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using perception::geometry::CameraDistortionModel;
using perception::geometry::CameraIntrinsics;
using perception::geometry::ResizeFit;
using perception::geometry::crop_intrinsics;
using perception::geometry::resize_offset;
using perception::geometry::resize_scale;
using perception::geometry::scale_intrinsics;
using perception::geometry::ImageSize;
using perception::geometry::resize_intrinsics;
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
}

TEST_F(CameraIntrinsicsTests, test_k_inverse_is_the_inverse_of_k) {
    const CameraIntrinsics intrinsics = shipped_left_intrinsics();
    const Eigen::Matrix3d product = intrinsics.K() * intrinsics.K_inv();
    EXPECT_TRUE(product.isApprox(Eigen::Matrix3d::Identity(), kEps));
}

TEST_F(CameraIntrinsicsTests, test_k_has_no_skew_term) {
    // Skew is not modelled: K_inv() is derived assuming K(0, 1) is zero, so
    // that slot staying empty is what makes the closed-form inverse correct.
    const CameraIntrinsics intrinsics = shipped_left_intrinsics();
    EXPECT_DOUBLE_EQ(intrinsics.K()(0, 1), 0.0);
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
    EXPECT_DOUBLE_EQ(intrinsics.to_row_major()[1], 0.0);
}

TEST_F(CameraIntrinsicsTests, test_from_row_major_reads_opencv_layout) {
    // Row-major [fx s cx; 0 fy cy; 0 0 1] -- a transpose here would read cx
    // out of the fy slot and go unnoticed on a square image, so every entry
    // is distinct.
    const CameraIntrinsics intrinsics = CameraIntrinsics::from_row_major(
        {100.0, 0.0, 320.0, 0.0, 200.0, 240.0, 0.0, 0.0, 1.0});

    EXPECT_DOUBLE_EQ(intrinsics.fx, 100.0);
    EXPECT_DOUBLE_EQ(intrinsics.cx, 320.0);
    EXPECT_DOUBLE_EQ(intrinsics.fy, 200.0);
    EXPECT_DOUBLE_EQ(intrinsics.cy, 240.0);
}

TEST_F(CameraIntrinsicsTests, test_from_row_major_rejects_skew) {
    // A sheared K means a pre-warped image or a bad calibration; the
    // rectification and disparity paths assume it away, so it is refused at
    // the boundary rather than silently dropped.
    EXPECT_THROW(CameraIntrinsics::from_row_major(
                     {100.0, 0.5, 320.0, 0.0, 200.0, 240.0, 0.0, 0.0, 1.0}),
                 std::runtime_error);
}

TEST_F(CameraIntrinsicsTests, test_rejects_non_positive_focals) {
    CameraIntrinsics intrinsics = shipped_left_intrinsics();
    intrinsics.fx = 0.0;
    EXPECT_THROW(intrinsics.validate_intrinsics(), std::runtime_error);

    // NOTE: K(), K_inv(), project_point() and backproject_ray() no longer call
    // validate_intrinsics() -- only rectified_to_source_pixel() does. So K_inv()
    // with fx = 0 returns inf rather than throwing. Asserted here as the
    // current contract; see the note in the review if that was not intended.
    EXPECT_NO_THROW(intrinsics.K());
    EXPECT_FALSE(std::isfinite(intrinsics.K_inv()(0, 0)));

    intrinsics.fx = 100.0;
    intrinsics.fy = -1.0;
    EXPECT_THROW(intrinsics.validate_intrinsics(), std::runtime_error);
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

    // Against the model's own documented contract, not a tighter number.
    // The iteration stops once a step moves the iterate less than
    // kUndistortTolerance, so that is what it promises; the old 1e-12 here
    // measured how far past the promise 40 unconditional iterations happened
    // to run. Worst observed on this rig is ~4.4e-11 normalized, which is
    // ~1e-7 px at this focal length.
    EXPECT_LT(worst, CameraDistortionModel::kUndistortTolerance)
        << "worst round trip " << worst;
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

class ResizeIntrinsicsTests : public ::testing::Test {
   public:
    // A 4:3 camera and the 5:3 grid a network wants it on. Cropping to fill it
    // scales by 960/1440 = 2/3 and takes 144 rows, 72 off each end.
    static constexpr ImageSize kSource{1440, 1080};
    static constexpr ImageSize kTarget{960, 576};

    static CameraIntrinsics camera() {
        return CameraIntrinsics{.fx = 2329.75, .fy = 2329.57, .cx = 754.5, .cy = 560.6};
    }
};

TEST_F(ResizeIntrinsicsTests, test_crop_takes_the_larger_ratio) {
    EXPECT_DOUBLE_EQ(resize_scale(kSource, kTarget, ResizeFit::Crop), 2.0 / 3.0);

    // 1080 * 2/3 = 720 against a 576 target: 144 rows over, 72 off each end.
    const Eigen::Vector2d offset = resize_offset(kSource, kTarget, ResizeFit::Crop);
    EXPECT_DOUBLE_EQ(offset.x(), 0.0);
    EXPECT_DOUBLE_EQ(offset.y(), 72.0);

    // Taller target: height is now the binding axis, and the crop moves to x.
    EXPECT_DOUBLE_EQ(resize_scale(kSource, ImageSize{960, 960}, ResizeFit::Crop),
                     960.0 / 1080.0);
}

TEST_F(ResizeIntrinsicsTests, test_pad_takes_the_smaller_ratio) {
    // 576/1080 fits the whole frame inside the target; 1440 * 0.5333 = 768
    // leaves 192 columns of pad, 96 a side. The offset is that pad, negative.
    EXPECT_DOUBLE_EQ(resize_scale(kSource, kTarget, ResizeFit::Pad), 576.0 / 1080.0);

    const Eigen::Vector2d offset = resize_offset(kSource, kTarget, ResizeFit::Pad);
    EXPECT_DOUBLE_EQ(offset.x(), -96.0);
    EXPECT_DOUBLE_EQ(offset.y(), 0.0);
}

// The two fits are one operation, and an exact aspect match is where that shows:
// nothing to crop is also nothing to pad, so they agree exactly.
TEST_F(ResizeIntrinsicsTests, test_the_fits_agree_when_the_aspect_matches) {
    for (const ImageSize target : {ImageSize{720, 540}, kSource}) {
        EXPECT_DOUBLE_EQ(resize_scale(kSource, target, ResizeFit::Crop),
                         resize_scale(kSource, target, ResizeFit::Pad));
        EXPECT_EQ(resize_offset(kSource, target, ResizeFit::Crop),
                  resize_offset(kSource, target, ResizeFit::Pad));
        EXPECT_DOUBLE_EQ(resize_offset(kSource, target, ResizeFit::Crop).x(), 0.0);
    }
    EXPECT_DOUBLE_EQ(resize_scale(kSource, ImageSize{720, 540}, ResizeFit::Crop), 0.5);
}

// The scale primitive on its own. Scaling about pixel centres rather than the
// grid origin is what makes it compose: two scales are one scale by the
// product, which a dropped half pixel breaks.
TEST_F(ResizeIntrinsicsTests, test_scale_alone_is_uniform_and_composes) {
    const CameraIntrinsics full = camera();

    const CameraIntrinsics half = scale_intrinsics(full, 0.5);
    EXPECT_DOUBLE_EQ(half.fx, 0.5 * full.fx);
    EXPECT_DOUBLE_EQ(half.fy, 0.5 * full.fy);
    EXPECT_DOUBLE_EQ(half.cx, 0.5 * (full.cx + 0.5) - 0.5);
    EXPECT_DOUBLE_EQ(half.cy, 0.5 * (full.cy + 0.5) - 0.5);

    const CameraIntrinsics twice = scale_intrinsics(scale_intrinsics(full, 0.5), 0.5);
    const CameraIntrinsics once = scale_intrinsics(full, 0.25);
    EXPECT_DOUBLE_EQ(twice.fx, once.fx);
    EXPECT_NEAR(twice.cx, once.cx, 1e-12);
    EXPECT_NEAR(twice.cy, once.cy, 1e-12);

    const CameraIntrinsics unchanged = scale_intrinsics(full, 1.0);
    EXPECT_DOUBLE_EQ(unchanged.cx, full.cx);
    EXPECT_DOUBLE_EQ(unchanged.cy, full.cy);

    EXPECT_THROW(scale_intrinsics(full, 0.0), std::runtime_error);
    EXPECT_THROW(scale_intrinsics(full, -0.5), std::runtime_error);
}

// The crop primitive on its own: a pure translation, so the focals do not move
// and no half pixel enters. Negative is a pad, which is the only thing that
// makes letterboxing the same operation as cropping.
TEST_F(ResizeIntrinsicsTests, test_crop_alone_only_moves_the_centre) {
    const CameraIntrinsics full = camera();

    const CameraIntrinsics cropped = crop_intrinsics(full, 100.0, 72.0);
    EXPECT_DOUBLE_EQ(cropped.fx, full.fx);
    EXPECT_DOUBLE_EQ(cropped.fy, full.fy);
    EXPECT_DOUBLE_EQ(cropped.cx, full.cx - 100.0);
    EXPECT_DOUBLE_EQ(cropped.cy, full.cy - 72.0);

    // Additive, and a pad is a crop the other way -- so one undoes the other.
    const CameraIntrinsics back = crop_intrinsics(cropped, -100.0, -72.0);
    EXPECT_DOUBLE_EQ(back.cx, full.cx);
    EXPECT_DOUBLE_EQ(back.cy, full.cy);

    EXPECT_THROW(crop_intrinsics(full, std::nan(""), 0.0), std::runtime_error);
}

// And the composition is the whole of resize_intrinsics(): scale, then crop by
// the offset. Stated as a test so the main function cannot drift from the two
// primitives it is meant to be made of.
TEST_F(ResizeIntrinsicsTests, test_resize_is_scale_then_crop) {
    for (const ResizeFit fit : {ResizeFit::Crop, ResizeFit::Pad}) {
        const Eigen::Vector2d offset = resize_offset(kSource, kTarget, fit);
        const CameraIntrinsics composed =
            crop_intrinsics(scale_intrinsics(camera(), resize_scale(kSource, kTarget, fit)),
                            offset.x(), offset.y());
        const CameraIntrinsics resized = resize_intrinsics(camera(), kSource, kTarget, fit);

        EXPECT_DOUBLE_EQ(resized.fx, composed.fx);
        EXPECT_DOUBLE_EQ(resized.fy, composed.fy);
        EXPECT_DOUBLE_EQ(resized.cx, composed.cx);
        EXPECT_DOUBLE_EQ(resized.cy, composed.cy);
    }
}

TEST_F(ResizeIntrinsicsTests, test_focals_scale_and_the_centre_moves_with_the_crop) {
    const CameraIntrinsics resized =
        resize_intrinsics(camera(), kSource, kTarget, ResizeFit::Crop);
    const double s = 2.0 / 3.0;

    EXPECT_DOUBLE_EQ(resized.fx, s * camera().fx);
    EXPECT_DOUBLE_EQ(resized.fy, s * camera().fy);

    // cx' = s * (cx + 0.5) - 0.5, no horizontal crop.
    EXPECT_DOUBLE_EQ(resized.cx, s * (camera().cx + 0.5) - 0.5);
    // cy' carries the 72-row crop on top of the same scaling.
    EXPECT_DOUBLE_EQ(resized.cy, s * (camera().cy + 0.5) - 0.5 - 72.0);

    EXPECT_NO_THROW(resized.validate_intrinsics());
}

// Letterboxing moves the principal point *inwards* by the pad, where a crop
// moves it outwards -- the same subtraction, the offset's sign doing the work.
TEST_F(ResizeIntrinsicsTests, test_letterbox_insets_the_principal_point) {
    const CameraIntrinsics resized =
        resize_intrinsics(camera(), kSource, kTarget, ResizeFit::Pad);
    const double s = 576.0 / 1080.0;

    EXPECT_DOUBLE_EQ(resized.fx, s * camera().fx);
    EXPECT_DOUBLE_EQ(resized.cx, s * (camera().cx + 0.5) - 0.5 + 96.0);
    EXPECT_DOUBLE_EQ(resized.cy, s * (camera().cy + 0.5) - 0.5);
    EXPECT_NO_THROW(resized.validate_intrinsics());

    // The source's left edge lands at target column 96, so a pixel in the pad
    // band projects to a negative source coordinate: nothing was sampled there.
    EXPECT_DOUBLE_EQ(resize_offset(kSource, kTarget, ResizeFit::Pad).x(), -96.0);
}

TEST_F(ResizeIntrinsicsTests, test_resizing_to_the_same_size_changes_nothing) {
    const CameraIntrinsics resized =
        resize_intrinsics(camera(), kSource, kSource, ResizeFit::Crop);

    EXPECT_DOUBLE_EQ(resized.fx, camera().fx);
    EXPECT_DOUBLE_EQ(resized.fy, camera().fy);
    EXPECT_DOUBLE_EQ(resized.cx, camera().cx);
    EXPECT_DOUBLE_EQ(resized.cy, camera().cy);
}

// The half pixel is the whole reason this is a helper rather than a
// multiplication at each call site: projecting a point through the resized
// intrinsics must land where the resize maps the full-resolution projection,
// and a dropped +0.5 misses by half a pixel times (1 - s) everywhere at once.
TEST_F(ResizeIntrinsicsTests, test_projection_agrees_with_the_pixel_mapping) {
    const CameraIntrinsics full = camera();

    for (const ResizeFit fit : {ResizeFit::Crop, ResizeFit::Pad}) {
        const CameraIntrinsics resized = resize_intrinsics(full, kSource, kTarget, fit);
        const double s = resize_scale(kSource, kTarget, fit);
        const Eigen::Vector2d offset = resize_offset(kSource, kTarget, fit);

        for (const Eigen::Vector2d& normalized :
             {Eigen::Vector2d{0.0, 0.0}, Eigen::Vector2d{0.21, -0.13},
              Eigen::Vector2d{-0.3, 0.27}}) {
            const Eigen::Vector2d in_full = full.normalized_to_pixel(normalized);
            const Eigen::Vector2d in_target = resized.normalized_to_pixel(normalized);

            // Forward: the same scale-and-offset the intrinsics encode.
            EXPECT_NEAR(in_target.x(), s * (in_full.x() + 0.5) - 0.5 - offset.x(), 1e-9);
            EXPECT_NEAR(in_target.y(), s * (in_full.y() + 0.5) - 0.5 - offset.y(), 1e-9);

            // And back again, undoing the scale and offset by hand.
            const Eigen::Vector2d round_trip{
                (in_target.x() + offset.x() + 0.5) / s - 0.5,
                (in_target.y() + offset.y() + 0.5) / s - 0.5};
            EXPECT_TRUE(VectorsNear(round_trip, in_full, 1e-9));
        }
    }
}

TEST_F(ResizeIntrinsicsTests, test_rejects_an_unset_size) {
    EXPECT_THROW(resize_scale(kSource, ImageSize{}, ResizeFit::Crop), std::runtime_error);
    EXPECT_THROW(resize_intrinsics(camera(), ImageSize{}, kTarget, ResizeFit::Pad),
                 std::runtime_error);
    EXPECT_THROW(resize_offset(kSource, ImageSize{960, 0}, ResizeFit::Crop),
                 std::runtime_error);
}

class RectificationTests : public ::testing::Test {
   public:
    RectificationTests() = default;
    void SetUp() override {}
};

TEST_F(RectificationTests, test_from_row_major_reads_opencv_layout) {
    const Rectification rectification = shipped_left_rectification();

    EXPECT_DOUBLE_EQ(rectification.rotation(0, 1), 0.09624063867688702);
    EXPECT_DOUBLE_EQ(rectification.rotation(1, 0), -0.09624479980195409);
    EXPECT_DOUBLE_EQ(rectification.rectified_intrinsics().fx,
                     2619.5697059803665);
    EXPECT_DOUBLE_EQ(rectification.projection(0, 2), 745.2493591308594);
    EXPECT_DOUBLE_EQ(rectification.baseline_term(), 0.0);
}

TEST_F(RectificationTests, test_rectifying_rotations_agree_across_the_pair) {
    // The invariant that matters: R2 * R == R1, both eyes landing in the SAME
    // rectified orientation. Bouguet splits R evenly between them, so R1 and
    // R2 each carry half of it away from a common levelling roll.
    const Eigen::Matrix3d R1 = shipped_left_rectification().rotation;

    Eigen::Matrix3d R2;
    R2 << 0.9961549149424362, 0.08760449966380064, 0.0009149178556072229,
        -0.08760177054709829, 0.996151949769823, -0.0026875205661547707,
        -0.001146836100276492, 0.0025970383969276044, 0.9999959700711417;

    Eigen::Matrix3d R;  // left -> right, from stereoCalibrate
    R << 0.9999622862355996, 0.008678895253704862, 0.00032075480980638674,
        -0.008680474272847443, 0.9999482680785398, 0.005301936733629774,
        -0.00027472326299085706, -0.00530452108151135, 0.9999858931921113;

    EXPECT_TRUE((R2 * R).isApprox(R1, 1e-12));
}

class RectifiedToSourcePixelTests : public ::testing::Test {
   public:
    RectifiedToSourcePixelTests() = default;
    void SetUp() override {}
};

TEST_F(RectifiedToSourcePixelTests, test_identity_calibration_is_identity) {
    // K == P, R == I, no distortion: every pixel must map to itself. A dropped
    // term anywhere shows up here as an offset.
    const CameraIntrinsics intrinsics{
        .fx = 1050.0, .fy = 1050.0, .cx = 720.0, .cy = 540.0};
    const CameraDistortionModel distortion;

    Rectification rectification;
    rectification.rotation = Eigen::Matrix3d::Identity();
    rectification.projection << 1050.0, 0.0, 720.0, 0.0, 0.0, 1050.0, 540.0,
        0.0, 0.0, 0.0, 1.0, 0.0;

    for (double v : {0.0, 539.0, 1079.0}) {
        for (double u : {0.0, 719.0, 1439.0}) {
            const auto source = rectified_to_source_pixel(
                intrinsics, distortion, rectification, u, v);
            ASSERT_TRUE(source.has_value());
            EXPECT_NEAR(source->x(), u, 1e-9);
            EXPECT_NEAR(source->y(), v, 1e-9);
        }
    }
}

TEST_F(RectifiedToSourcePixelTests, test_round_trips_through_the_calibration) {
    // Follow the map to the source pixel, then push that pixel forward by hand
    // -- undistort, rotate into the rectified frame, project with P -- and land
    // back where we started. This is what a transposed R or a distortion sign
    // error breaks.
    const CameraIntrinsics intrinsics = shipped_left_intrinsics();
    const CameraDistortionModel distortion = shipped_left_distortion();
    const Rectification rectification = shipped_left_rectification();

    double worst = 0.0;
    for (double v = 40.0; v < 1080.0; v += 97.0) {
        for (double u = 40.0; u < 1440.0; u += 91.0) {
            const auto source = rectified_to_source_pixel(
                intrinsics, distortion, rectification, u, v);
            ASSERT_TRUE(source.has_value());

            const Eigen::Vector2d normalized{
                (source->x() - intrinsics.cx) / intrinsics.fx,
                (source->y() - intrinsics.cy) / intrinsics.fy};
            const Eigen::Vector2d undistorted =
                distortion.undistort_normalized(normalized);

            const Eigen::Vector3d ray =
                rectification.rotation *
                Eigen::Vector3d{undistorted.x(), undistorted.y(), 1.0};
            const Eigen::Vector2d reprojected{
                rectification.projection(0, 0) * (ray.x() / ray.z()) +
                    rectification.projection(0, 2),
                rectification.projection(1, 1) * (ray.y() / ray.z()) +
                    rectification.projection(1, 2)};

            worst = std::max(worst, std::abs(reprojected.x() - u));
            worst = std::max(worst, std::abs(reprojected.y() - v));
        }
    }
    EXPECT_LT(worst, 1e-3) << "worst round-trip error " << worst << " px";
}

TEST_F(RectifiedToSourcePixelTests, test_baseline_column_is_ignored) {
    // P(0,3) is the OTHER eye's baseline offset. If it moved this eye's map,
    // every rectified frame would shift by a baseline's worth of pixels.
    const CameraIntrinsics intrinsics = shipped_left_intrinsics();
    const CameraDistortionModel distortion = shipped_left_distortion();

    Rectification without = shipped_left_rectification();
    Rectification with = shipped_left_rectification();
    with.projection(0, 3) = -284.86842219049794;

    const auto a =
        rectified_to_source_pixel(intrinsics, distortion, without, 700.0, 500.0);
    const auto b =
        rectified_to_source_pixel(intrinsics, distortion, with, 700.0, 500.0);
    ASSERT_TRUE(a.has_value() && b.has_value());
    EXPECT_EQ(a->x(), b->x());
    EXPECT_EQ(a->y(), b->y());
}

TEST_F(RectifiedToSourcePixelTests, test_rejects_singular_projection) {
    const CameraIntrinsics intrinsics = shipped_left_intrinsics();
    const CameraDistortionModel distortion = shipped_left_distortion();

    Rectification rectification = shipped_left_rectification();
    rectification.projection.setZero();

    EXPECT_THROW(rectified_to_source_pixel(intrinsics, distortion,
                                           rectification, 0.0, 0.0),
                 std::runtime_error);
}
