// Behaviour tests for the rectification lookup map. Builds maps on the host and
// checks their entries; no GPU is touched, but the map builder lives in
// perception_processing, which only exists in a CUDA build.
//
// The properties checked here are the ones that go wrong silently: an identity
// camera that is not identity, a transpose in the rotation, a missing half
// pixel, distortion applied in the wrong direction. All of them still produce a
// plausible-looking image.
#include "transforms/rectify_map.hpp"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++failures;
}

using perception::build_rectify_map;
using perception::count_out_of_frame;
using perception::RectifyMapCoords;
using perception::geometry::CameraDistortionModel;
using perception::geometry::CameraIntrinsics;
using perception::geometry::ImageSize;
using perception::geometry::PinholeCameraModel;
using perception::geometry::Rectification;

constexpr uint32_t kWidth = 1440;
constexpr uint32_t kHeight = 1080;

// Source and rectified are the same grid here, which is the case the shipped
// calibration is. They are still passed separately: build_rectify_map divides
// Normalized entries by the source extent and sizes the table by the rectified
// one, and conflating them is exactly what these tests would not catch.
constexpr ImageSize kSize{kWidth, kHeight};

// The shipped left camera, copied from app/config/stereo_calibration.yaml. Not
// loaded from the file: this test is about the map, and a schema change should
// break the loader's test, not this one.
// The eye and its half of the rectification, which no longer travel in one
// struct: the camera is mono calibration, the rectification belongs to the pair.
struct Eye {
  PinholeCameraModel camera;
  Rectification rectification;
};

Eye shipped_left() {
  Eye eye;
  PinholeCameraModel& cal = eye.camera;
  cal.intrinsics = CameraIntrinsics::from_row_major(
      {2329.7529233353694, 0.0, 754.5022389652534,
       0.0, 2329.5740433167261, 560.6093817146035,
       0.0, 0.0, 1.0});
  cal.distortion = CameraDistortionModel::from_coefficients(
      {-0.24410080138920398, 0.26696399429948375, 8.827990083798891e-05,
       2.4006501080748617e-05, -0.15345901438237441});
  eye.rectification = Rectification::from_row_major(
      {0.9953566462359332, 0.09624063867688702, 0.001698899944114407,
       -0.09624479980195409, 0.9953543863342791, 0.0025659532716845275,
       -0.0014440585296353384, -0.0027175489279068757, 0.9999952648001825},
      {2619.5697059803665, 0.0, 745.2493591308594, 0.0,
       0.0, 2619.5697059803665, 558.5332374572754, 0.0,
       0.0, 0.0, 1.0, 0.0});
  return eye;
}

// Same geometry, but nothing to undo: K == P, R == I, no distortion.
Eye identity_camera() {
  Eye eye;
  PinholeCameraModel& cal = eye.camera;
  cal.intrinsics = CameraIntrinsics::from_row_major(
      {1050.0, 0.0, 720.0, 0.0, 1050.0, 540.0, 0.0, 0.0, 1.0});
  cal.distortion = CameraDistortionModel{};
  eye.rectification = Rectification::from_row_major(
      {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
      {1050.0, 0.0, 720.0, 0.0, 0.0, 1050.0, 540.0, 0.0, 0.0, 0.0, 1.0, 0.0});
  return eye;
}

const float2& at(const std::vector<float2>& map, uint32_t x, uint32_t y) {
  return map[static_cast<std::size_t>(y) * kWidth + x];
}

void test_identity_is_identity() {
  const Eye eye = identity_camera();
  const std::vector<float2> map =
      build_rectify_map(eye.camera, kSize, eye.rectification, kSize, RectifyMapCoords::Pixels);

  check(map.size() == static_cast<std::size_t>(kWidth) * kHeight, "identity: one entry per pixel");

  // Every pixel samples its own texel centre: index i at coordinate i + 0.5.
  // Dropping that half pixel shifts the whole image by half a texel, which
  // looks like nothing and blurs everything.
  double worst = 0.0;
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      const float2 m = at(map, x, y);
      worst = std::max(worst, std::abs(static_cast<double>(m.x) - (x + 0.5)));
      worst = std::max(worst, std::abs(static_cast<double>(m.y) - (y + 0.5)));
    }
  }
  check(worst < 1e-3, "identity: map is the half-pixel identity (worst " + std::to_string(worst) +
                          "px)");
  check(count_out_of_frame(map, kWidth, kHeight, RectifyMapCoords::Pixels) == 0,
        "identity: nothing samples outside the frame");
}

void test_normalized_matches_pixels() {
  const Eye cal = shipped_left();
  const std::vector<float2> pixels =
      build_rectify_map(cal.camera, kSize, cal.rectification, kSize, RectifyMapCoords::Pixels);
  const std::vector<float2> normalized =
      build_rectify_map(cal.camera, kSize, cal.rectification, kSize, RectifyMapCoords::Normalized);

  double worst = 0.0;
  for (std::size_t i = 0; i < pixels.size(); ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(normalized[i].x) * kWidth - pixels[i].x));
    worst = std::max(worst, std::abs(static_cast<double>(normalized[i].y) * kHeight - pixels[i].y));
  }
  check(worst < 1e-2, "coords: normalized is pixels / size (worst " + std::to_string(worst) + "px)");
}

// The real test of the maths: take a rectified pixel, follow the map to the
// source pixel, then push that source pixel forward through the calibration by
// hand -- undistort it and project it into the rectified frame -- and land back
// where we started. This is the round trip that a transposed rotation or a
// distortion sign error breaks.
void test_round_trip_through_calibration() {
  const Eye cal = shipped_left();
  const std::vector<float2> map =
      build_rectify_map(cal.camera, kSize, cal.rectification, kSize, RectifyMapCoords::Pixels);

  const auto& K = cal.camera.intrinsics;
  const auto& R = cal.rectification.rotation;
  const auto& P = cal.rectification.projection;
  const double k1 = cal.camera.distortion.k1, k2 = cal.camera.distortion.k2;
  const double p1 = cal.camera.distortion.p1, p2 = cal.camera.distortion.p2,
               k3 = cal.camera.distortion.k3;

  double worst = 0.0;
  for (uint32_t y = 40; y < kHeight; y += 97) {
    for (uint32_t x = 40; x < kWidth; x += 91) {
      const float2 m = at(map, x, y);

      // Source pixel -> normalised distorted coordinates.
      const double xd = (m.x - 0.5 - K.cx) / K.fx;
      const double yd = (m.y - 0.5 - K.cy) / K.fy;

      // Undistort by fixed-point iteration on the forward model, which is what
      // cv::undistortPoints does. The forward model is what the map applied, so
      // converging here means the map's distortion is the same distortion.
      double ux = xd, uy = yd;
      for (int i = 0; i < 40; ++i) {
        const double r2 = ux * ux + uy * uy;
        const double radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
        const double dx = 2.0 * p1 * ux * uy + p2 * (r2 + 2.0 * ux * ux);
        const double dy = p1 * (r2 + 2.0 * uy * uy) + 2.0 * p2 * ux * uy;
        ux = (xd - dx) / radial;
        uy = (yd - dy) / radial;
      }

      // Rotate into the rectified frame and project with P.
      const double rx = R(0, 0) * ux + R(0, 1) * uy + R(0, 2);
      const double ry = R(1, 0) * ux + R(1, 1) * uy + R(1, 2);
      const double rz = R(2, 0) * ux + R(2, 1) * uy + R(2, 2);
      const double u = P(0, 0) * (rx / rz) + P(0, 2);
      const double v = P(1, 1) * (ry / rz) + P(1, 2);

      worst = std::max(worst, std::abs(u - x));
      worst = std::max(worst, std::abs(v - y));
    }
  }
  check(worst < 1e-3, "shipped left: rectified -> source -> rectified round trips (worst " +
                          std::to_string(worst) + "px)");
}

// stereoRectify was run at alpha=0 and reports a full-frame valid ROI for both
// eyes, so no rectified pixel should need a source pixel that does not exist.
// If this ever fires, the map is fine and the calibration changed.
void test_shipped_maps_stay_in_frame() {
  for (const char* role : {"left", "right"}) {
    Eye cal = shipped_left();
    if (std::string(role) == "right") {
      cal.camera.intrinsics = CameraIntrinsics::from_row_major(
          {2325.7999086770888, 0.0, 735.2969373843101,
           0.0, 2325.69699359287, 559.0702576161585,
           0.0, 0.0, 1.0});
      cal.camera.distortion = CameraDistortionModel::from_coefficients(
          {-0.23956893837514504, 0.21113576451839627, 3.624182815897604e-05,
           -1.877730621643997e-05, 0.09839628999662872});
      cal.rectification = Rectification::from_row_major(
          {0.9961549149424362, 0.08760449966380064, 0.0009149178556072229,
           -0.08760177054709829, 0.996151949769823, -0.0026875205661547707,
           -0.001146836100276492, 0.0025970383969276044, 0.9999959700711417},
          {2619.5697059803665, 0.0, 745.2493591308594, -284.86842219049794,
           0.0, 2619.5697059803665, 558.5332374572754, 0.0,
           0.0, 0.0, 1.0, 0.0});
    }
    const std::vector<float2> map =
        build_rectify_map(cal.camera, kSize, cal.rectification, kSize, RectifyMapCoords::Pixels);
    const std::size_t outside = count_out_of_frame(map, kWidth, kHeight, RectifyMapCoords::Pixels);
    check(outside == 0, std::string("shipped ") + role + ": " + std::to_string(outside) +
                            " entries outside the frame, expected 0");
  }
}

// P2[3] is the other eye's baseline offset. It must not move this eye's map --
// if it does, every rectified frame is shifted by a baseline's worth of pixels.
void test_baseline_column_is_ignored() {
  Eye cal = shipped_left();
  const std::vector<float2> without =
      build_rectify_map(cal.camera, kSize, cal.rectification, kSize, RectifyMapCoords::Pixels);
  cal.rectification.projection(0, 3) = -284.86842219049794;
  const std::vector<float2> with =
      build_rectify_map(cal.camera, kSize, cal.rectification, kSize, RectifyMapCoords::Pixels);

  double worst = 0.0;
  for (std::size_t i = 0; i < with.size(); ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(with[i].x) - without[i].x));
    worst = std::max(worst, std::abs(static_cast<double>(with[i].y) - without[i].y));
  }
  check(worst == 0.0, "P[3] does not enter this eye's map");
}

// The two distortion-model rejections used to live in build_rectify_map, which
// was the only thing that read the raw coefficient vector. They now happen
// earlier -- from_coefficients() when the numbers enter the type system, and
// load_stereo_calibration() when the model name is read -- so an unsupported
// model is refused at parse rather than at the first map build. Both are
// checked where they now are, so that moving them did not quietly drop them.
void test_rejects_bad_input() {
  bool threw = false;
  try {
    const Eye eye = identity_camera();
    build_rectify_map(eye.camera, kSize, eye.rectification, ImageSize{0, kHeight},
                      RectifyMapCoords::Pixels);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "zero width is rejected");

  // Hoisted out of the per-pixel maths and into the map builder, so it is
  // checked once per map instead of 1.5M times -- but still checked, because
  // an fx of 0 would otherwise fill every entry with the principal point.
  threw = false;
  try {
    Eye cal = identity_camera();
    cal.camera.intrinsics.fx = 0.0;
    build_rectify_map(cal.camera, kSize, cal.rectification, kSize, RectifyMapCoords::Pixels);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "a zero focal length is rejected before the loop runs");

  threw = false;
  try {
    // Eight coefficients is OpenCV's rational model.
    CameraDistortionModel::from_coefficients(std::vector<double>(8, 0.1));
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "an eight-coefficient rational model is rejected, not truncated");

  threw = false;
  try {
    Eye cal = identity_camera();
    cal.rectification.projection.setZero();
    build_rectify_map(cal.camera, kSize, cal.rectification, kSize, RectifyMapCoords::Pixels);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "a singular projection is rejected rather than producing NaNs");
}

}  // namespace

int main() {
  test_identity_is_identity();
  test_normalized_matches_pixels();
  test_round_trip_through_calibration();
  test_shipped_maps_stay_in_frame();
  test_baseline_column_is_ignored();
  test_rejects_bad_input();

  std::printf("%s\n", failures == 0 ? "all passed" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
