#include "stereo_calibration.hpp"

#include <yaml-cpp/yaml.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace perception {
namespace {

[[noreturn]] void fail(const std::string& where, const std::string& what) {
  throw std::runtime_error("calibration: " + where + ": " + what);
}

const YAML::Node require_node(const YAML::Node& parent, const char* key, const std::string& where) {
  const YAML::Node node = parent[key];
  if (!node) fail(where, std::string("missing '") + key + "'");
  return node;
}

// Fixed-length matrices are read into std::array so a wrong count is caught
// here, naming the key, rather than as an out-of-range read much later.
template <std::size_t N>
std::array<double, N> read_matrix(const YAML::Node& parent, const char* key,
                                  const std::string& where) {
  const YAML::Node node = require_node(parent, key, where);
  const std::string at = where + "." + key;
  if (!node.IsSequence()) fail(at, "expected a flat row-major sequence");
  if (node.size() != N) {
    fail(at, "expected " + std::to_string(N) + " values row-major, got " +
                 std::to_string(node.size()));
  }

  std::array<double, N> out{};
  for (std::size_t i = 0; i < N; ++i) {
    try {
      out[i] = node[i].as<double>();
    } catch (const YAML::Exception&) {
      fail(at, "element " + std::to_string(i) + " is not a number");
    }
  }
  return out;
}

CameraCalibration read_camera(const YAML::Node& node, const std::string& where) {
  CameraCalibration cal;

  cal.role = require_node(node, "role", where).as<std::string>();
  if (node["serial"]) cal.serial = node["serial"].as<std::string>();

  // The file stays flat and row-major; the geometry types are built from it
  // here so that this is the only place the layout is interpreted.
  cal.intrinsics = geometry::CameraIntrinsics::from_row_major(
      read_matrix<9>(node, "camera_matrix", where));

  const YAML::Node distortion = require_node(node, "distortion", where);
  if (distortion["model"]) cal.distortion_model = distortion["model"].as<std::string>();
  if (cal.distortion_model != "plumb_bob") {
    fail(where + ".distortion.model",
         "'" + cal.distortion_model +
             "' is not supported; this build implements plumb_bob "
             "[k1 k2 p1 p2 k3] only");
  }

  const YAML::Node coefficients = require_node(distortion, "coefficients", where + ".distortion");
  if (!coefficients.IsSequence()) {
    fail(where + ".distortion.coefficients", "expected a sequence");
  }
  std::vector<double> raw;
  for (std::size_t i = 0; i < coefficients.size(); ++i) {
    raw.push_back(coefficients[i].as<double>());
  }
  try {
    cal.distortion = geometry::CameraDistortionModel::from_coefficients(raw);
  } catch (const std::runtime_error& e) {
    fail(where + ".distortion.coefficients", e.what());
  }

  const YAML::Node rect = require_node(node, "rectification", where);
  cal.rectification = geometry::Rectification::from_row_major(
      read_matrix<9>(rect, "rotation", where + ".rectification"),
      read_matrix<12>(rect, "projection", where + ".rectification"));

  if (cal.intrinsics.fx <= 0.0 || cal.intrinsics.fy <= 0.0) {
    fail(where + ".camera_matrix", "fx and fy must be positive");
  }
  return cal;
}

}  // namespace

double StereoCalibration::rectified_baseline_m() const {
  // Sign dropped: the baseline is a distance, and which eye is left is the
  // roles' business. baseline() owns the -P2(0,3) / fx convention, and throws
  // on a projection it cannot divide through -- the same stance as the loader
  // above, where an approximate calibration is worse than none.
  return std::abs(camera[1].rectification.baseline());
}

std::string StereoCalibration::summary() const {
  char line[256];
  std::snprintf(line, sizeof(line),
                "calibration: %ux%u, %s|%s, baseline=%.4fm (rectified %.4fm), fx=%.2fpx",
                width, height, camera[0].role.c_str(), camera[1].role.c_str(), baseline_m,
                rectified_baseline_m(), rectified_fx());
  return line;
}

StereoCalibration load_stereo_calibration(const std::string& path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("calibration: cannot load " + path + ": " + e.what());
  }

  StereoCalibration cal;

  const YAML::Node size = require_node(root, "image_size", "");
  cal.width = require_node(size, "width", "image_size").as<uint32_t>();
  cal.height = require_node(size, "height", "image_size").as<uint32_t>();
  if (cal.width == 0 || cal.height == 0) fail("image_size", "must be non-zero");

  const YAML::Node cameras = require_node(root, "cameras", "");
  if (!cameras.IsSequence() || cameras.size() != 2) {
    fail("cameras", "expected exactly two entries, one per stream, in stream order");
  }
  for (std::size_t s = 0; s < 2; ++s) {
    cal.camera[s] = read_camera(cameras[s], "cameras[" + std::to_string(s) + "]");
  }
  if (cal.camera[0].role == cal.camera[1].role) {
    fail("cameras", "both entries have role '" + cal.camera[0].role +
                        "'; the two eyes must be distinguishable");
  }

  const YAML::Node extrinsics = require_node(root, "extrinsics", "");
  const std::array<double, 9> r = read_matrix<9>(extrinsics, "rotation", "extrinsics");
  const std::array<double, 3> t = read_matrix<3>(extrinsics, "translation_m", "extrinsics");
  cal.rotation << r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8];
  cal.translation_m << t[0], t[1], t[2];

  cal.baseline_m = require_node(root, "baseline_m", "").as<double>();
  if (cal.baseline_m <= 0.0) fail("baseline_m", "must be positive");

  // The file states the baseline twice -- once as |T| and once spelled out.
  // Disagreement means someone edited one and not the other, and a wrong
  // baseline scales every depth in the scene without looking wrong.
  const double norm = cal.translation_m.norm();
  if (std::abs(norm - cal.baseline_m) > 1e-4) {
    char detail[192];
    std::snprintf(detail, sizeof(detail),
                  "baseline_m is %.6fm but |extrinsics.translation_m| is %.6fm; they describe "
                  "the same distance and must agree",
                  cal.baseline_m, norm);
    fail("baseline_m", detail);
  }

  const std::array<double, 16> q = read_matrix<16>(
      require_node(root, "rectification", ""), "disparity_to_depth", "rectification");
  cal.disparity_to_depth << q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10],
      q[11], q[12], q[13], q[14], q[15];

  // Same check against the rectified projections, which is where depth is
  // actually computed from -- a Q or a P that was not regenerated alongside the
  // extrinsics is the failure this catches.
  const double rectified = cal.rectified_baseline_m();
  if (rectified > 0.0 && std::abs(rectified - cal.baseline_m) > 1e-3) {
    char detail[224];
    std::snprintf(detail, sizeof(detail),
                  "the rectified projections imply a %.6fm baseline but baseline_m is %.6fm; "
                  "rectification and extrinsics came from different calibration runs",
                  rectified, cal.baseline_m);
    fail("rectification", detail);
  }

  return cal;
}

}  // namespace perception
