#include "app_config.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <unordered_map>

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

std::string scalar(const YAML::Node& node, const std::string& where) {
  if (!node.IsScalar()) fail(where, "expected a scalar value");
  return node.Scalar();
}

ReuseWait readReuseWait(const std::string& value, const std::string& where) {
  if (value == "HostSync") return ReuseWait::HostSync;
  if (value == "DeviceWait") return ReuseWait::DeviceWait;
  fail(where, "expected HostSync or DeviceWait, got '" + value + "'");
}

WritePolicy readWritePolicy(const std::string& value, const std::string& where) {
  if (value == "RoundRobin") return WritePolicy::RoundRobin;
  if (value == "ScanForFree") return WritePolicy::ScanForFree;
  fail(where, "expected RoundRobin or ScanForFree, got '" + value + "'");
}

}  // namespace

AppConfig load_app_config(const std::string& path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("config: cannot load " + path + ": " + e.what());
  }

  AppConfig config;

  // The camera section belongs to spinnaker/, which parses it with its own
  // loader so the standalone driver and this app cannot drift apart.
  config.camera = load_camera_config(path);

  if (const YAML::Node pipeline = root["pipeline"]) {
    read(pipeline, "ingress_depth", "pipeline", config.pipeline.ingress_depth);
    read(pipeline, "device_depth", "pipeline", config.pipeline.device_depth);
    read(pipeline, "max_consumers", "pipeline", config.pipeline.max_consumers);
    read(pipeline, "device_id", "pipeline", config.pipeline.device_id);

    if (pipeline["reuse_wait"]) {
      config.pipeline.reuse_wait = readReuseWait(
          scalar(pipeline["reuse_wait"], "pipeline.reuse_wait"), "pipeline.reuse_wait");
    }
    if (pipeline["write_policy"]) {
      config.pipeline.write_policy = readWritePolicy(
          scalar(pipeline["write_policy"], "pipeline.write_policy"), "pipeline.write_policy");
    }
  }

  if (const YAML::Node upload = root["upload"]) {
    read(upload, "scratch_slots", "upload", config.upload.scratch_slots);
    read(upload, "use_graph", "upload", config.upload.use_graph);

    if (upload["pop_timeout_ms"]) {
      config.upload.pop_timeout =
          std::chrono::milliseconds(require<int64_t>(upload["pop_timeout_ms"],
                                                     "upload.pop_timeout_ms"));
    }
  }

  if (const YAML::Node display = root["display"]) {
    read(display, "enable", "display", config.display.enable);
    read(display, "window_width", "display", config.display.window_width);
    read(display, "window_height", "display", config.display.window_height);
    read(display, "vsync", "display", config.display.vsync);
    read(display, "latency_scale_ms", "display", config.display.latency_scale_ms);
  }

  // The upload stage allocates on this device, so it has to be the one the
  // rings were built on.
  config.upload.device_id = config.pipeline.device_id;
  return config;
}

ImageDesc to_image_desc(const CameraGeometry& geometry) {
  static const std::unordered_map<std::string, PixelFormat> kFormats = {
      {"BayerRG8", PixelFormat::Bayer8_RGGB}, {"BayerGR8", PixelFormat::Bayer8_GRBG},
      {"BayerGB8", PixelFormat::Bayer8_GBRG}, {"BayerBG8", PixelFormat::Bayer8_BGGR},
      {"Mono8", PixelFormat::GRAY8},          {"RGB8", PixelFormat::RGB8},
      {"RGBa8", PixelFormat::RGBA8},
  };

  const auto it = kFormats.find(geometry.pixel_format);
  if (it == kFormats.end()) {
    throw std::runtime_error("unsupported camera pixel format '" + geometry.pixel_format + "'");
  }

  ImageDesc desc;
  desc.width = geometry.width;
  desc.height = geometry.height;
  desc.stride_bytes = geometry.stride_bytes;
  desc.format = it->second;
  return desc;
}

}  // namespace perception
