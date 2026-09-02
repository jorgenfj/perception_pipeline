#include "app_config.hpp"

#include "config_path.hpp"

#include <yaml-cpp/yaml.h>

#include <array>
#include <filesystem>
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

DisparityColormap readColormap(const std::string& value, const std::string& where) {
  if (value == "turbo") return DisparityColormap::Turbo;
  if (value == "jet") return DisparityColormap::Jet;
  fail(where, "expected turbo or jet, got '" + value + "'");
}

void readChannels(const YAML::Node& parent, const char* key, const std::string& where,
                  float (&out)[3]) {
  if (!parent[key]) return;
  const std::string at = where + "." + key;
  const auto values = require<std::vector<float>>(parent[key], at);
  if (values.size() != 3) {
    fail(at, "expected three values, one per channel, got " + std::to_string(values.size()));
  }
  for (int c = 0; c < 3; ++c) out[c] = values[c];
}

ViewerMode readViewerMode(const std::string& value, const std::string& where) {
  if (value == "camera") return ViewerMode::Camera;
  if (value == "yolo") return ViewerMode::Yolo;
  if (value == "headless") return ViewerMode::Headless;
  if (value == "ess") return ViewerMode::Ess;
  fail(where, "expected camera, yolo, headless, or ess, got '" + value + "'");
}

std::filesystem::path exe_dir() {
  std::error_code ec;
  const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  return ec ? std::filesystem::current_path() : exe.parent_path();
}

}  // namespace

const char* to_string(ViewerMode mode) {
  switch (mode) {
    case ViewerMode::Camera:
      return "camera";
    case ViewerMode::Yolo:
      return "yolo";
    case ViewerMode::Headless:
      return "headless";
    case ViewerMode::Ess:
      return "ess";
  }
  return "?";
}

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
  }

  if (root["viewer"]) {
    config.viewer_mode = readViewerMode(scalar(root["viewer"], "viewer"), "viewer");
  }

  if (const YAML::Node display = root["display"]) {
    read(display, "window_width", "display", config.display.window_width);
    read(display, "window_height", "display", config.display.window_height);
    read(display, "vsync", "display", config.display.vsync);
    read(display, "latency_scale_ms", "display", config.display.latency_scale_ms);
  }

  if (const YAML::Node yolo = root["yolo"]) {
    read(yolo, "engine_path", "yolo", config.yolo.engine_path);
    read(yolo, "model_width", "yolo", config.yolo.model_width);
    read(yolo, "model_height", "yolo", config.yolo.model_height);
    read(yolo, "conf_threshold", "yolo", config.yolo.conf_threshold);
  }

  if (const YAML::Node ess = root["ess"]) {
    read(ess, "enabled", "ess", config.ess.enabled);
    read(ess, "engine_path", "ess", config.ess.engine_path);
    read(ess, "plugin_path", "ess", config.ess.plugin_path);
    read(ess, "conf_threshold", "ess", config.ess.conf_threshold);
    read(ess, "display_min_disparity", "ess", config.ess.display_min_disparity);
    read(ess, "display_max_disparity", "ess", config.ess.display_max_disparity);
    if (ess["colormap"]) {
      config.ess.colormap = readColormap(scalar(ess["colormap"], "ess.colormap"), "ess.colormap");
    }
    if (const YAML::Node normalization = ess["normalization"]) {
      readChannels(normalization, "mean", "ess.normalization", config.ess.normalization.mean);
      readChannels(normalization, "stddev", "ess.normalization", config.ess.normalization.stddev);
      read(normalization, "no_source_value", "ess.normalization",
           config.ess.normalization.no_source_value);
    }
  }

  // Only consulted by a -DPERCEPTION_SOURCE=recording build, but always parsed:
  // one config file has to describe both, or switching source means editing the
  // config as well as reconfiguring and the two versions drift apart.
  if (const YAML::Node source = root["source"]) {
    read(source, "recording", "source", config.source.directory);
    read(source, "stream", "source", config.source.stream);
    read(source, "role", "source", config.source.role);
    read(source, "speed", "source", config.source.speed);
    read(source, "loop", "source", config.source.loop);
    read(source, "rebase_timestamps", "source", config.source.rebase_timestamps);
    read(source, "slot_wait_ms", "source", config.source.slot_wait_ms);
  }

  // Parsed by the camera schema, not here: stereo_view reads the same section,
  // and two parsers for one section is how they end up disagreeing.
  config.action_sync = load_action_sync_config(path);

  // --- streams ---------------------------------------------------------------
  // Absent means one stream, taking camera.serial. That is the pre-stereo
  // config verbatim, so an existing file keeps working unchanged.
  if (const YAML::Node streams = root["streams"]) {
    if (!streams.IsSequence() || streams.size() == 0) {
      fail("streams", "expected a non-empty sequence, one entry per camera");
    }
    if (streams.size() > 2) {
      fail("streams", "at most two streams; pairing more than a pair is a different problem");
    }
    for (std::size_t i = 0; i < streams.size(); ++i) {
      const std::string where = "streams[" + std::to_string(i) + "]";
      StreamConfig stream;
      read(streams[i], "role", where, stream.role);
      read(streams[i], "serial", where, stream.serial);
      if (stream.role.empty()) stream.role = "cam" + std::to_string(i);
      config.streams.push_back(stream);
    }
  }
  if (config.streams.empty()) {
    config.streams.push_back(StreamConfig{"cam0", config.camera.serial});
  }

  // --- stereo ----------------------------------------------------------------
  if (const YAML::Node stereo = root["stereo"]) {
    read(stereo, "enabled", "stereo", config.stereo.enabled);
    read(stereo, "reference_stream", "stereo", config.stereo.reference_stream);
    read(stereo, "calibration", "stereo", config.stereo.calibration_path);

    uint64_t tolerance_us = config.stereo.consumer.tolerance_ns / 1000;
    read(stereo, "tolerance_us", "stereo", tolerance_us);
    config.stereo.consumer.tolerance_ns = tolerance_us * 1000;

    read(stereo, "retry_attempts", "stereo", config.stereo.consumer.retry_attempts);

    uint64_t retry_ms = static_cast<uint64_t>(config.stereo.consumer.retry_wait.count());
    read(stereo, "retry_ms", "stereo", retry_ms);
    config.stereo.consumer.retry_wait = std::chrono::milliseconds(retry_ms);
  }

  if (config.stereo.enabled) {
    if (config.streams.size() != 2) {
      fail("stereo.enabled",
           "pairing needs exactly two streams, but `streams:` declares " +
               std::to_string(config.streams.size()));
    }
    if (config.stereo.reference_stream > 1) {
      fail("stereo.reference_stream", "must be 0 or 1");
    }
    if (config.pipeline.max_consumers < 3) {
      fail("pipeline.max_consumers",
           "stereo needs at least 3 (main loop, viewer, stereo consumer), got " +
               std::to_string(config.pipeline.max_consumers));
    }
    if (!config.stereo.calibration_path.empty()) {
      const std::string calibration_path =
          resolve_config_path(config.stereo.calibration_path,
                              std::filesystem::path(path).parent_path().string());
      config.calibration = load_stereo_calibration(calibration_path);
      config.have_calibration = true;

      const std::array<CalibrationIdentity, 2> identity =
          read_calibration_identity(calibration_path);
      for (std::size_t s = 0; s < 2; ++s) {
        const CalibrationIdentity& cal = identity[s];
        if (cal.role != config.streams[s].role) {
          fail("stereo.calibration",
               "cameras[" + std::to_string(s) + "] is role '" + cal.role +
                   "' but streams[" + std::to_string(s) + "] is '" + config.streams[s].role +
                   "'; the calibration would be applied to the wrong eye");
        }
        if (!cal.serial.empty() && !config.streams[s].serial.empty() &&
            cal.serial != config.streams[s].serial) {
          fail("stereo.calibration",
               "cameras[" + std::to_string(s) + "] was taken on camera " + cal.serial +
                   " but streams[" + std::to_string(s) + "] opens " + config.streams[s].serial);
        }
      }
    }
  }

  if (config.ess.enabled) {
    if (!config.stereo.enabled) {
      fail("ess.enabled", "needs stereo.enabled -- disparity comes from a pair, not one stream");
    }
    if (!config.have_calibration) {
      fail("ess.enabled", "needs stereo.calibration -- the rectify maps are built from it");
    }
    if (config.ess.engine_path.empty()) {
      fail("ess.engine_path", "empty, but ess.enabled is set");
    }
    if (config.ess.plugin_path.empty()) {
      fail("ess.plugin_path",
           "empty, but ess.enabled is set -- every NGC ESS export is built from fused ops that "
           "only exist in the plugin library, and the engine will not deserialize without it");
    }
    if (!(config.ess.display_max_disparity > config.ess.display_min_disparity)) {
      fail("ess.display_max_disparity", "must be greater than ess.display_min_disparity");
    }
  }
  if (config.viewer_mode == ViewerMode::Ess && !config.ess.enabled) {
    fail("viewer", "'ess' is the disparity window; set ess.enabled to run the engine behind it");
  }

  config.upload.device_id = config.pipeline.device_id;
  return config;
}

// Compiled in by app/CMakeLists.txt; see capture/include/config_path.hpp for
// why the config is found rather than copied into bin/.
#ifndef PERCEPTION_CONFIG_DIR
#define PERCEPTION_CONFIG_DIR ""
#endif

std::string default_config_path() {
  return resolve_config_path("acquire.yaml", PERCEPTION_CONFIG_DIR);
}

std::string resolve_next_to_exe(const std::string& path) {
  if (path.empty() || std::filesystem::path(path).is_absolute()) return path;
  return (exe_dir() / path).string();
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
