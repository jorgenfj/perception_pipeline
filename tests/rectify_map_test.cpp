// Behaviour tests for the rectification lookup map. Builds maps on the host and
// checks their entries; the builder is plain camera maths in
// perception_geometry, so no GPU and no CUDA build is involved.
//
// The properties checked here are the ones that go wrong silently: an identity
// camera that is not identity, a transpose in the rotation, a missing half
// pixel, distortion applied in the wrong direction. All of them still produce a
// plausible-looking image.
#include <perception/geometry/stereo.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace {

int failures = 0;

// For a condition asserted inside a sampling loop: only the first failure is
// worth printing, and a pass says nothing at all.
void check_once(bool ok, const std::string& what) {
  static std::string reported;
  if (ok || reported == what) return;
  reported = what;
  std::printf("FAIL  %s\n", what.c_str());
  ++failures;
}

void check(bool ok, const std::string& what) {
  std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++failures;
}

using perception::geometry::build_rectify_map;
using perception::geometry::CameraDistortionModel;
using perception::geometry::CameraIntrinsics;
using perception::geometry::ImageSize;
using perception::geometry::PinholeCameraModel;
using perception::geometry::ResizeFit;
using perception::geometry::resize_offset;
using perception::geometry::resize_scale;
using perception::geometry::Rectification;
using perception::geometry::StereoRectification;
using perception::geometry::rectified_to_source_pixel;
using perception::geometry::resize_rectification;

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

// The shipped right eye, same source.
Eye shipped_right() {
  Eye eye;
  eye.camera.intrinsics = CameraIntrinsics::from_row_major(
      {2325.7999086770888, 0.0, 735.2969373843101,
       0.0, 2325.69699359287, 559.0702576161585,
       0.0, 0.0, 1.0});
  eye.camera.distortion = CameraDistortionModel::from_coefficients(
      {-0.23956893837514504, 0.21113576451839627, 3.624182815897604e-05,
       -1.877730621643997e-05, 0.09839628999662872});
  eye.rectification = Rectification::from_row_major(
      {0.9961549149424362, 0.08760449966380064, 0.0009149178556072229,
       -0.08760177054709829, 0.996151949769823, -0.0026875205661547707,
       -0.001146836100276492, 0.0025970383969276044, 0.9999959700711417},
      {2619.5697059803665, 0.0, 745.2493591308594, -284.86842219049794,
       0.0, 2619.5697059803665, 558.5332374572754, 0.0,
       0.0, 0.0, 1.0, 0.0});
  return eye;
}

// Both halves of the shipped stereoRectify run, with its Q. The extrinsics are
// left out: nothing here needs them, and resize_rectification() validates the rectification on
// its own terms.
StereoRectification shipped_rectification() {
  StereoRectification rectification;
  rectification.cameras[0] = shipped_left().rectification;
  rectification.cameras[1] = shipped_right().rectification;
  rectification.size = kSize;
  rectification.disparity_to_depth <<
      1.0, 0.0, 0.0, -745.2493591308594,
      0.0, 1.0, 0.0, -558.5332374572754,
      0.0, 0.0, 0.0, 2619.5697059803665,
      0.0, 0.0, 9.195718099735886, 0.0;
  return rectification;
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

// The map is a flat float vector, two per pixel: (u, v) at 2 * (y * width + x).
struct Entry {
  float x, y;
};

// How many of a map's entries point outside the source image. The library used
// to offer this; the pipeline assumes a cropped fit where the count is zero, so
// it lives here, where the assumption is what is being checked.
std::size_t count_outside(const std::vector<float>& map, uint32_t source_width,
                          uint32_t source_height) {
  std::size_t count = 0;
  for (std::size_t i = 0; i + 1 < map.size(); i += 2) {
    if (map[i] < 0.0f || map[i] >= source_width ||  //
        map[i + 1] < 0.0f || map[i + 1] >= source_height) {
      ++count;
    }
  }
  return count;
}

// Where a pixel of the target grid sits on the calibrated one -- the inverse of
// what resize_rectification() folds into the intrinsics, spelled out here rather than taken
// from geometry so these checks derive it independently.
Eigen::Vector2d to_calibrated(ImageSize source, ImageSize target, double u, double v) {
  const double scale = resize_scale(source, target, ResizeFit::Crop);
  const Eigen::Vector2d offset = resize_offset(source, target, ResizeFit::Crop);
  return {(u + offset.x() + 0.5) / scale - 0.5, (v + offset.y() + 0.5) / scale - 0.5};
}

Entry at_size(const std::vector<float>& map, uint32_t x, uint32_t y, uint32_t width) {
  const std::size_t i = 2 * (static_cast<std::size_t>(y) * width + x);
  return {map[i], map[i + 1]};
}

Entry at(const std::vector<float>& map, uint32_t x, uint32_t y) {
  return at_size(map, x, y, kWidth);
}

void test_identity_is_identity() {
  const Eye eye = identity_camera();
  const std::vector<float> map =
      build_rectify_map(eye.camera, eye.rectification, kSize);

  check(map.size() == 2 * static_cast<std::size_t>(kWidth) * kHeight,
        "identity: two floats per pixel");

  // Every pixel samples its own texel centre: index i at coordinate i + 0.5.
  // Dropping that half pixel shifts the whole image by half a texel, which
  // looks like nothing and blurs everything.
  double worst = 0.0;
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      const Entry m = at(map, x, y);
      worst = std::max(worst, std::abs(static_cast<double>(m.x) - (x + 0.5)));
      worst = std::max(worst, std::abs(static_cast<double>(m.y) - (y + 0.5)));
    }
  }
  check(worst < 1e-3, "identity: map is the half-pixel identity (worst " + std::to_string(worst) +
                          "px)");
  check(count_outside(map, kWidth, kHeight) == 0,
        "identity: nothing samples outside the frame");
}

// The map is in source pixels, always. Normalizing is EssPreprocessTransform's
// job now -- it is the only thing that knows what its texture wants -- so the
// old "normalized is pixels / size" check has no host-side home and lives on
// only as the transform's own staging loop.

// The real test of the maths: take a rectified pixel, follow the map to the
// source pixel, then push that source pixel forward through the calibration by
// hand -- undistort it and project it into the rectified frame -- and land back
// where we started. This is the round trip that a transposed rotation or a
// distortion sign error breaks.
void test_round_trip_through_calibration() {
  const Eye cal = shipped_left();
  const std::vector<float> map =
      build_rectify_map(cal.camera, cal.rectification, kSize);

  const auto& K = cal.camera.intrinsics;
  const auto& R = cal.rectification.rotation;
  const auto& P = cal.rectification.projection;
  const double k1 = cal.camera.distortion.k1, k2 = cal.camera.distortion.k2;
  const double p1 = cal.camera.distortion.p1, p2 = cal.camera.distortion.p2,
               k3 = cal.camera.distortion.k3;

  double worst = 0.0;
  for (uint32_t y = 40; y < kHeight; y += 97) {
    for (uint32_t x = 40; x < kWidth; x += 91) {
      const Entry m = at(map, x, y);

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
    const Eye cal = std::string(role) == "right" ? shipped_right() : shipped_left();
    const std::vector<float> map =
        build_rectify_map(cal.camera, cal.rectification, kSize);
    const std::size_t outside = count_outside(map, kWidth, kHeight);
    check(outside == 0, std::string("shipped ") + role + ": " + std::to_string(outside) +
                            " entries outside the frame, expected 0");
  }
}

// P2[3] is the other eye's baseline offset. It must not move this eye's map --
// if it does, every rectified frame is shifted by a baseline's worth of pixels.
void test_baseline_column_is_ignored() {
  Eye cal = shipped_left();
  const std::vector<float> without =
      build_rectify_map(cal.camera, cal.rectification, kSize);
  cal.rectification.projection(0, 3) = -284.86842219049794;
  const std::vector<float> with =
      build_rectify_map(cal.camera, cal.rectification, kSize);

  double worst = 0.0;
  for (std::size_t i = 0; i < with.size(); ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(with[i]) - without[i]));
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
    build_rectify_map(eye.camera, eye.rectification, ImageSize{0, kHeight});
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
    build_rectify_map(cal.camera, cal.rectification, kSize);
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
}

// --- resizing onto the network's grid ----------------------------------------
// ESS's full model takes 960x576. The calibrated grid is 1440x1080, so the fit
// scales by 960/1440 and crops the excess rows, and the resize is folded into
// the rectified intrinsics rather than done as a second resample.
constexpr uint32_t kEssWidth = 960;
constexpr uint32_t kEssHeight = 576;
constexpr ImageSize kEssSize{kEssWidth, kEssHeight};

void test_the_fit_scales_by_width() {
  const double scale = resize_scale(kSize, kEssSize, ResizeFit::Crop);
  check(std::abs(scale - 960.0 / 1440.0) < 1e-12,
        "crop fit: the larger ratio (" + std::to_string(scale) + ")");

  // 1080 * 2/3 = 720, and 720 - 576 = 144 rows, half off each end. The crop is
  // not returned anywhere, so it is read back through the inverse: target pixel
  // (0, 0) is the centre of the target's first pixel, which sits at target
  // (0, 72) before the crop and so at (0.5, 72.5) / scale - 0.5 on the
  // calibrated grid.
  const Eigen::Vector2d origin = to_calibrated(kSize, kEssSize, 0.0, 0.0);
  check(std::abs(origin.x() - (0.5 / scale - 0.5)) < 1e-9,
        "crop fit: nothing comes off horizontally (" + std::to_string(origin.x()) + ")");
  check(std::abs(origin.y() - ((72.0 + 0.5) / scale - 0.5)) < 1e-9,
        "crop fit: 72 rows off the top (" + std::to_string(origin.y()) + ")");

  // Same aspect in and out means nothing to crop, and no scale is a no-op --
  // including the half pixel, which a wrong convention breaks precisely here.
  check(resize_scale(kSize, kSize, ResizeFit::Crop) == 1.0, "crop fit: an identical target is the identity");
  const Eigen::Vector2d same = to_calibrated(kSize, kSize, 17.0, 23.0);
  check(std::abs(same.x() - 17.0) < 1e-12 && std::abs(same.y() - 23.0) < 1e-12,
        "crop fit: an identical target maps a pixel to itself");
}

// The point of folding the resize into the intrinsics: a map built for the
// network's grid must sample the *same source pixel* the full-resolution
// pipeline would have sampled at the corresponding place. If it does not, the
// the resize has introduced a shift -- a half-pixel one if the +0.5 convention was
// dropped, which is exactly the kind of error that still looks like a picture.
//
// Checked against rectified_to_source_pixel() on the *unscaled* rectification
// rather than against a full-resolution map, because a target pixel almost
// never lands on a full-resolution pixel centre: at 2/3 scale it falls on
// quarter pixels, so there is no entry to compare with, only a coordinate.
void test_resize_map_matches_full_resolution() {
  const Eye cal = shipped_left();
  const Rectification scaled = resize_rectification(cal.rectification, kSize, kEssSize, ResizeFit::Crop);

  const std::vector<float> small =
      build_rectify_map(cal.camera, scaled, kEssSize);

  check(small.size() == 2 * static_cast<std::size_t>(kEssWidth) * kEssHeight,
        "resize: the map is sized for the network's grid");

  double worst = 0.0;
  for (uint32_t y = 6; y < kEssHeight; y += 17) {
    for (uint32_t x = 3; x < kEssWidth; x += 13) {
      // Where this target pixel's centre sits on the calibrated grid, undoing
      // the crop and the scale in pixel-index convention.
      const Eigen::Vector2d full_px = to_calibrated(kSize, kEssSize, x, y);

      const std::optional<Eigen::Vector2d> expected =
          rectified_to_source_pixel(cal.camera.intrinsics, cal.camera.distortion,
                                    cal.rectification, full_px.x(), full_px.y());
      check_once(expected.has_value(), "resize: the shipped map has a source pixel everywhere");

      const std::size_t i = 2 * (static_cast<std::size_t>(y) * kEssWidth + x);
      // The map entry carries the texel-centre half pixel; the raw coordinate
      // does not.
      worst = std::max(worst, std::abs(static_cast<double>(small[i]) - (expected->x() + 0.5)));
      worst = std::max(worst, std::abs(static_cast<double>(small[i + 1]) - (expected->y() + 0.5)));
    }
  }
  check(worst < 1e-2, "resize: the small map samples where the full one does (worst " +
                          std::to_string(worst) + "px)");
}

// Letterboxing the same pair: the whole field of view survives at a smaller
// scale, and the price is a pad band that no source pixel maps into. The map
// still has an entry for every pixel of it -- entries pointing outside the
// source, which is what the kernel writes its pad value into.
void test_letterbox_map_pads_rather_than_crops() {
  const Eye cal = shipped_left();
  const Rectification scaled = resize_rectification(cal.rectification, kSize, kEssSize, ResizeFit::Pad);

  const std::vector<float> map =
      build_rectify_map(cal.camera, scaled, kEssSize);

  const double scale = resize_scale(kSize, kEssSize, ResizeFit::Pad);
  check(std::abs(scale - 576.0 / 1080.0) < 1e-12,
        "letterbox: scale fits the whole frame (" + std::to_string(scale) + ")");

  // 1440 * 0.5333 = 768 wide inside a 960 target, so 96 columns a side fall
  // outside the calibrated rectified grid. They are NOT all sourceless: the
  // shipped rectification is alpha=0, which crops the rectified frame well
  // inside what the source image actually covers, so a rectified coordinate
  // just past the frame still lands on a real pixel. Letterboxing here buys
  // back some of the field of view alpha=0 threw away, and only the outermost
  // part of the band is true pad.
  const std::size_t outside = count_outside(map, kWidth, kHeight);
  const std::size_t band = 2 * 96 * static_cast<std::size_t>(kEssHeight);
  check(outside > 0 && outside < band,
        "letterbox: " + std::to_string(outside) + " of " + std::to_string(band) +
            " pad-band entries have no source pixel");

  // The image band itself is the alpha=0 grid rescaled, so every entry in it
  // samples the source -- the letterbox added pad, it did not push image out.
  std::size_t inside_band_outside_source = 0;
  for (uint32_t y = 0; y < kEssHeight; ++y) {
    for (uint32_t x = 96; x < kEssWidth - 96; ++x) {
      const Entry m = at_size(map, x, y, kEssWidth);
      if (m.x < 0.0f || m.x >= kWidth || m.y < 0.0f || m.y >= kHeight) {
        ++inside_band_outside_source;
      }
    }
  }
  check(inside_band_outside_source == 0,
        "letterbox: the image band samples entirely within the source (" +
            std::to_string(inside_band_outside_source) + " outside)");

  // The outer edge of the band, on the row where the rectifying roll pushes it
  // furthest past the source, is real pad. (The band's corners are not: the
  // 5.5deg roll in R1 means "outside the rectified frame" and "outside the
  // source image" are not the same region, which is why the count above is a
  // range and not a rectangle's worth of pixels.)
  const Entry outer = at_size(map, 0, kEssHeight / 2, kEssWidth);
  check(outer.x < 0.0f || outer.x >= kWidth,
        "letterbox: the outer pad column has no source pixel (" + std::to_string(outer.x) + ")");

  // Crop would have kept 2/3 rather than 0.533, which is the trade: 25% more
  // disparity resolution for 13% of the vertical field of view.
  check(scale < resize_scale(kSize, kEssSize, ResizeFit::Crop),
        "letterbox: keeps the field of view at a smaller scale than cropping");
}

// The rig does not change because the grid did. fx scales, P(0,3) scales with
// it, and the baseline they imply -- the thing depth is computed from -- comes
// out untouched.
void test_resize_preserves_the_rig() {
  const StereoRectification full = shipped_rectification();
  const StereoRectification scaled = resize_rectification(full, kEssSize, ResizeFit::Crop);
  const double scale = resize_scale(full.size, kEssSize, ResizeFit::Crop);

  check(std::abs(scaled.baseline_m() - full.baseline_m()) < 1e-9,
        "resize: the baseline is unchanged (" + std::to_string(scaled.baseline_m()) + "m)");
  check(std::abs(scaled.rectified_fx() - scale * full.rectified_fx()) < 1e-9,
        "resize: fx scales with the grid");
  check(scaled.size == kEssSize, "resize: the pair states the grid it is now for");

  // Depth is fx * baseline / disparity, and disparity scales by exactly the
  // same factor as fx, so a disparity measured on the network's grid gives the
  // same metres as the full-resolution one. That invariance is the whole reason
  // a caller can work in ESS pixels without unscaling first.
  const double disparity_full = 64.0;
  const double depth_full =
      full.rectified_fx() * full.baseline_m() / disparity_full;
  const double depth_small =
      scaled.rectified_fx() * scaled.baseline_m() / (disparity_full * scale);
  check(std::abs(depth_full - depth_small) < 1e-9,
        "resize: depth is the same in ESS pixels (" + std::to_string(depth_small) + "m)");

  // Q must move with the intrinsics, or reprojecting in target pixels lands
  // somewhere else than the same point at full resolution.
  const double u = 480.0, v = 300.0, d = disparity_full * scale;
  const Eigen::Vector4d small_point =
      scaled.disparity_to_depth * Eigen::Vector4d(u, v, d, 1.0);
  const Eigen::Vector2d full_px = to_calibrated(full.size, kEssSize, u, v);
  const Eigen::Vector4d full_point =
      full.disparity_to_depth *
      Eigen::Vector4d(full_px.x(), full_px.y(), disparity_full, 1.0);
  const Eigen::Vector3d a = small_point.hnormalized();
  const Eigen::Vector3d b = full_point.hnormalized();
  check((a - b).norm() < 1e-6, "resize: Q reprojects to the same metric point (" +
                                   std::to_string((a - b).norm()) + "m apart)");
}

// A crop deep enough to push the principal point off the grid produces a
// rectification that cannot describe the images it claims to, and resize_rectification()
// validates rather than handing it back.
void test_resize_rejects_a_degenerate_crop() {
  const StereoRectification full = shipped_rectification();
  bool threw = false;
  try {
    resize_rectification(full, ImageSize{960, 16}, ResizeFit::Crop);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "resize: a crop that loses the principal point is rejected");
}

}  // namespace

int main() {
  test_identity_is_identity();
  test_round_trip_through_calibration();
  test_shipped_maps_stay_in_frame();
  test_baseline_column_is_ignored();
  test_rejects_bad_input();
  test_the_fit_scales_by_width();
  test_letterbox_map_pads_rather_than_crops();
  test_resize_map_matches_full_resolution();
  test_resize_preserves_the_rig();
  test_resize_rejects_a_degenerate_crop();

  std::printf("%s\n", failures == 0 ? "all passed" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
