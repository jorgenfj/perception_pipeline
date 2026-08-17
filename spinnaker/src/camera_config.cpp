#include "camera_config.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

namespace perception {
namespace {

[[noreturn]] void fail(const std::string& where, const std::string& what) {
  throw std::runtime_error("config: " + where + ": " + what);
}

// yaml-cpp reports a type mismatch as an exception with no key context, which
// makes a typo in a long file hard to place. Everything goes through here so
// the message always names the key.
template <typename T>
T require(const YAML::Node& node, const std::string& where) {
  try {
    return node.as<T>();
  } catch (const YAML::Exception&) {
    fail(where, "cannot read '" + YAML::Dump(node) + "' as the expected type");
  }
}

template <typename T>
void read(const YAML::Node& parent, const char* key, const std::string& where, T& out) {
  if (!parent[key]) return;
  out = require<T>(parent[key], where + "." + key);
}

// Scalars stay as text: the GenICam node's own type decides how to parse them
// later, and going through YAML's type inference here would turn "Off" into a
// boolean and 8000 into something a float node cannot take.
std::string scalar(const YAML::Node& node, const std::string& where) {
  if (!node.IsScalar()) fail(where, "expected a scalar value");
  return node.Scalar();
}

// Two spellings, because map key order is not something YAML guarantees and
// camera features genuinely depend on order:
//
//   features:                     features:
//     - PixelFormat: BayerRG8       PixelFormat: BayerRG8
//     - Width: 2048                 Width: 2048
//
// The sequence form is explicit about it; the map form is accepted because it
// reads better and yaml-cpp preserves document order in practice.
FeatureList readFeatures(const YAML::Node& node, const std::string& where) {
  FeatureList out;
  if (!node) return out;

  if (node.IsSequence()) {
    for (std::size_t i = 0; i < node.size(); ++i) {
      const YAML::Node& entry = node[i];
      const std::string at = where + "[" + std::to_string(i) + "]";
      if (!entry.IsMap() || entry.size() != 1) {
        fail(at, "each entry must be a single 'Node: value' pair");
      }
      const auto it = entry.begin();
      const std::string name = it->first.Scalar();
      out.emplace_back(name, scalar(it->second, at + "." + name));
    }
    return out;
  }

  if (node.IsMap()) {
    for (const auto& entry : node) {
      const std::string name = entry.first.Scalar();
      out.emplace_back(name, scalar(entry.second, where + "." + name));
    }
    return out;
  }

  fail(where, "expected a map or a sequence of single-key maps");
}

YAML::Node loadFile(const std::string& path) {
  try {
    return YAML::LoadFile(path);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("config: cannot load " + path + ": " + e.what());
  }
}

}  // namespace

CameraConfig load_camera_config(const std::string& path) {
  const YAML::Node root = loadFile(path);
  CameraConfig config;

  const YAML::Node camera = root["camera"];
  if (!camera) return config;

  read(camera, "serial", "camera", config.serial);
  read(camera, "timeout_ms", "camera", config.timeout_ms);
  config.features = readFeatures(camera["features"], "camera.features");
  config.stream_features = readFeatures(camera["stream_features"], "camera.stream_features");
  return config;
}

StandaloneConfig load_standalone_config(const std::string& path) {
  const YAML::Node root = loadFile(path);
  StandaloneConfig config;

  const YAML::Node standalone = root["standalone"];
  if (!standalone) return config;

  read(standalone, "buffer_count", "standalone", config.buffer_count);
  read(standalone, "max_frames", "standalone", config.max_frames);
  return config;
}

}  // namespace perception
