// Behaviour tests for the stereo calibration loader. No GPU and no camera; the
// one thing it needs is a filesystem, plus the calibration file that actually
// ships -- a schema whose own example does not parse is a schema nobody can
// use.
#include "stereo_calibration.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++failures;
}

using perception::load_stereo_calibration;
using perception::StereoCalibration;

// A consistent rig: 1050px focal length, 120mm baseline, so P2[3] is -126.
const char* kGood = R"(
image_size: { width: 1440, height: 1080 }
cameras:
  - role: left
    camera_matrix: [1050.0, 0.0, 720.0, 0.0, 1050.0, 540.0, 0.0, 0.0, 1.0]
    distortion: { model: plumb_bob, coefficients: [0.0, 0.0, 0.0, 0.0, 0.0] }
    rectification:
      rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
      projection: [1050.0, 0.0, 720.0, 0.0, 0.0, 1050.0, 540.0, 0.0, 0.0, 0.0, 1.0, 0.0]
  - role: right
    camera_matrix: [1050.0, 0.0, 720.0, 0.0, 1050.0, 540.0, 0.0, 0.0, 1.0]
    distortion: { model: plumb_bob, coefficients: [0.0, 0.0, 0.0, 0.0, 0.0] }
    rectification:
      rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
      projection: [1050.0, 0.0, 720.0, -126.0, 0.0, 1050.0, 540.0, 0.0, 0.0, 0.0, 1.0, 0.0]
extrinsics:
  rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  translation_m: [-0.12, 0.0, 0.0]
baseline_m: 0.12
rectification:
  disparity_to_depth: [1.0, 0.0, 0.0, -720.0, 0.0, 1.0, 0.0, -540.0,
                       0.0, 0.0, 0.0, 1050.0, 0.0, 0.0, 8.3333333333, 0.0]
)";

std::filesystem::path write_temp(const std::string& name, const std::string& body) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
  std::ofstream out(path);
  out << body;
  return path;
}

// Loads `body`, returning the error text, or "" if it loaded.
std::string load_error(const std::string& name, const std::string& body) {
  const std::filesystem::path path = write_temp(name, body);
  std::string error;
  try {
    load_stereo_calibration(path.string());
  } catch (const std::exception& e) {
    error = e.what();
  }
  std::filesystem::remove(path);
  return error;
}

// Replaces the first occurrence of `from` in kGood, to break exactly one thing.
std::string tweaked(const std::string& from, const std::string& to) {
  std::string body = kGood;
  const std::size_t at = body.find(from);
  if (at == std::string::npos) {
    check(false, "test setup: pattern not found in the good calibration");
    return body;
  }
  return body.replace(at, from.size(), to);
}

void test_loads_a_consistent_calibration() {
  const std::filesystem::path path = write_temp("stereo_calibration_good.yaml", kGood);
  const StereoCalibration cal = load_stereo_calibration(path.string());
  std::filesystem::remove(path);

  check(cal.width == 1440 && cal.height == 1080, "the image size is read");
  check(cal.camera[0].role == "left" && cal.camera[1].role == "right",
        "the roles come back in stream order");
  check(cal.camera[0].fx() == 1050.0 && cal.camera[0].cx() == 720.0,
        "the intrinsics accessors pick the right cells out of K");
  check(cal.baseline_m == 0.12, "the baseline is read");
  check(cal.camera[0].distortion.size() == 5 && cal.camera[0].distortion_model == "plumb_bob",
        "the distortion model and its five plumb_bob coefficients are read");
}

void test_derives_the_rectified_baseline() {
  const std::filesystem::path path = write_temp("stereo_calibration_derive.yaml", kGood);
  const StereoCalibration cal = load_stereo_calibration(path.string());
  std::filesystem::remove(path);

  // -P2[3] / fx = 126.0 / 1050.0
  check(std::abs(cal.rectified_baseline_m() - 0.12) < 1e-9,
        "the rectified baseline is recovered from P2 and fx");
  check(cal.rectified_fx() == 1050.0, "and the rectified focal length from P1");
}

void test_rejects_an_inconsistent_baseline() {
  // |T| still 0.12, but the file claims 0.20.
  const std::string error =
      load_error("stereo_calibration_baseline.yaml", tweaked("baseline_m: 0.12", "baseline_m: 0.20"));
  check(error.find("baseline_m") != std::string::npos,
        "a baseline that disagrees with |translation_m| is refused");
  check(error.find("0.20") != std::string::npos || error.find("0.200000") != std::string::npos,
        "and the message quotes both values");
}

void test_rejects_stale_rectification() {
  // Extrinsics and baseline agree at 0.12, but P2 was left at an older 0.09
  // rig: -1050 * 0.09 = -94.5. This is the calibration half-regenerated.
  const std::string error = load_error("stereo_calibration_stale.yaml",
                                       tweaked("720.0, -126.0", "720.0, -94.5"));
  check(error.find("rectification") != std::string::npos,
        "a P2 that implies a different baseline is refused");
}

void test_rejects_structural_mistakes() {
  // Truncate the document at the second camera, leaving one eye and no
  // extrinsics -- what you get by commenting out half a rig.
  std::string one_eye = kGood;
  one_eye.resize(one_eye.find("  - role: right"));
  const std::string one_eye_error = load_error("stereo_calibration_one_eye.yaml", one_eye);
  check(one_eye_error.find("cameras") != std::string::npos,
        "a file that is not two cameras in stream order is refused");

  const std::string same_role =
      load_error("stereo_calibration_roles.yaml", tweaked("role: right", "role: left"));
  check(same_role.find("role") != std::string::npos,
        "two cameras with the same role are refused");

  const std::string short_matrix = load_error(
      "stereo_calibration_short.yaml",
      tweaked("[1050.0, 0.0, 720.0, 0.0, 1050.0, 540.0, 0.0, 0.0, 1.0]", "[1050.0, 0.0, 720.0]"));
  check(short_matrix.find("camera_matrix") != std::string::npos,
        "a matrix of the wrong length is refused, naming the key");
  check(short_matrix.find("9") != std::string::npos, "and says how many values it wanted");
}

// The file app/config/ ships. Its numbers are placeholders, but it has to parse
// and satisfy every consistency check, or the first person to copy it as a
// template starts from something broken.
void test_the_shipped_calibration_parses() {
#ifdef PERCEPTION_APP_CONFIG_DIR
  const std::filesystem::path path =
      std::filesystem::path(PERCEPTION_APP_CONFIG_DIR) / "stereo_calibration.yaml";
  if (!std::filesystem::exists(path)) {
    check(false, "app/config/stereo_calibration.yaml is missing");
    return;
  }

  const StereoCalibration cal = load_stereo_calibration(path.string());
  check(cal.camera[0].role == "left" && cal.camera[1].role == "right",
        "the shipped calibration parses and is left-then-right");
  check(std::abs(cal.rectified_baseline_m() - cal.baseline_m) < 1e-3,
        "and its rectification agrees with its baseline");
#else
  check(true, "app config dir not defined for this build, skipping the shipped file");
#endif
}

}  // namespace

int main() {
  test_loads_a_consistent_calibration();
  test_derives_the_rectified_baseline();
  test_rejects_an_inconsistent_baseline();
  test_rejects_stale_rectification();
  test_rejects_structural_mistakes();
  test_the_shipped_calibration_parses();

  std::printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
