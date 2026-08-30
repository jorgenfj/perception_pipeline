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

geometry::PinholeCameraModel read_camera(const YAML::Node& node, const std::string& where) {
  geometry::PinholeCameraModel camera;

  // The file stays flat and row-major; the geometry types are built from it
  // here so that this is the only place the layout is interpreted.
  // from_row_major() validates K, so a non-positive or non-finite focal is
  // refused before it can reach anything.
  camera.intrinsics = geometry::CameraIntrinsics::from_row_major(
      read_matrix<9>(node, "camera_matrix", where));

  // The model name is checked and dropped rather than stored: only plumb_bob
  // parses, so a field holding it would be an invariant dressed as data.
  const YAML::Node distortion = require_node(node, "distortion", where);
  const std::string model =
      distortion["model"] ? distortion["model"].as<std::string>() : "plumb_bob";
  if (model != "plumb_bob") {
    fail(where + ".distortion.model",
         "'" + model +
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
    camera.distortion = geometry::CameraDistortionModel::from_coefficients(raw);
  } catch (const std::runtime_error& e) {
    fail(where + ".distortion.coefficients", e.what());
  }

  return camera;
}

geometry::ImageSize read_image_size(const YAML::Node& node, const std::string& where) {
  geometry::ImageSize size;
  size.width = require_node(node, "width", where).as<uint32_t>();
  size.height = require_node(node, "height", where).as<uint32_t>();
  if (size.empty()) fail(where, "must be non-zero");
  return size;
}

// The `cameras:` sequence, checked for the shape both readers depend on.
const YAML::Node require_cameras(const YAML::Node& root) {
  const YAML::Node cameras = require_node(root, "cameras", "");
  if (!cameras.IsSequence() || cameras.size() != 2) {
    fail("cameras", "expected exactly two entries, one per stream, in stream order");
  }
  return cameras;
}

// Roles are not kept on StereoCalibration -- the index is the eye -- but the
// file still has to be coherent about them, and this is the layer that reads
// the file. Two entries claiming the same eye is a copy-paste, not a rig.
void require_distinct_roles(const YAML::Node& cameras) {
  const std::string first = require_node(cameras[0], "role", "cameras[0]").as<std::string>();
  const std::string second = require_node(cameras[1], "role", "cameras[1]").as<std::string>();
  if (first == second) {
    fail("cameras", "both entries have role '" + first +
                        "'; the two eyes must be distinguishable");
  }
}

// The file keeps each eye's R and P under its camera, because that is how
// stereoRectify hands them back. They are read out into the pair-level type
// here: this is the one place the file's layout is interpreted, and grouping
// by eye is the file's business, not the calibration's.
geometry::Rectification read_rectification(const YAML::Node& node, const std::string& where) {
  const YAML::Node rect = require_node(node, "rectification", where);
  try {
    return geometry::Rectification::from_row_major(
        read_matrix<9>(rect, "rotation", where + ".rectification"),
        read_matrix<12>(rect, "projection", where + ".rectification"));
  } catch (const std::runtime_error& e) {
    fail(where + ".rectification", e.what());
  }
}

YAML::Node load_document(const std::string& path) {
  try {
    return YAML::LoadFile(path);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("calibration: cannot load " + path + ": " + e.what());
  }
}

}  // namespace

std::array<CalibrationIdentity, 2> read_calibration_identity(const std::string& path) {
  const YAML::Node cameras = require_cameras(load_document(path));
  require_distinct_roles(cameras);

  std::array<CalibrationIdentity, 2> identity;
  for (std::size_t s = 0; s < 2; ++s) {
    const std::string where = "cameras[" + std::to_string(s) + "]";
    identity[s].role = require_node(cameras[s], "role", where).as<std::string>();
    if (cameras[s]["serial"]) identity[s].serial = cameras[s]["serial"].as<std::string>();
  }
  return identity;
}

geometry::StereoCalibration load_stereo_calibration(const std::string& path) {
  const YAML::Node root = load_document(path);

  geometry::StereoCalibration cal;

  cal.size = read_image_size(require_node(root, "image_size", ""), "image_size");

  const YAML::Node cameras = require_cameras(root);
  require_distinct_roles(cameras);
  for (std::size_t s = 0; s < 2; ++s) {
    const std::string where = "cameras[" + std::to_string(s) + "]";
    cal.cameras[s] = read_camera(cameras[s], where);
    cal.rectification.cameras[s] = read_rectification(cameras[s], where);
  }

  // The file spells the baseline out on its own, away from the extrinsics
  // node; StereoExtrinsics takes all three together because checking them
  // against each other is the only reason to hold them apart.
  const YAML::Node extrinsics = require_node(root, "extrinsics", "");
  try {
    cal.extrinsics = geometry::StereoExtrinsics::from_row_major(
        read_matrix<9>(extrinsics, "rotation", "extrinsics"),
        read_matrix<3>(extrinsics, "translation_m", "extrinsics"),
        require_node(root, "baseline_m", "").as<double>());
  } catch (const std::runtime_error& e) {
    fail("baseline_m", e.what());
  }

  const YAML::Node rectification = require_node(root, "rectification", "");

  // stereoRectify's newImageSize. Absent means it was left at the default,
  // which is the size the cameras were calibrated at -- the only case any
  // calibration in this tree has needed so far, and the reason the key is
  // optional rather than required.
  cal.rectification.size =
      rectification["image_size"]
          ? read_image_size(rectification["image_size"], "rectification.image_size")
          : cal.size;

  const std::array<double, 16> q =
      read_matrix<16>(rectification, "disparity_to_depth", "rectification");
  cal.rectification.disparity_to_depth << q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8],
      q[9], q[10], q[11], q[12], q[13], q[14], q[15];

  // Both statements the geometry types own: that the pair is internally one
  // stereoRectify run, and that it is the run belonging to these extrinsics. A
  // Q or a P left behind when the extrinsics were regenerated is what the
  // second catches.
  try {
    cal.rectification.validate();
    cal.rectification.validate_against(cal.extrinsics);
  } catch (const std::exception& e) {
    fail("rectification", e.what());
  }

  return cal;
}

}  // namespace perception
