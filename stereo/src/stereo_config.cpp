#include "stereo_config.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

namespace perception {
namespace {

[[noreturn]] void fail(const std::string& where, const std::string& what) {
  throw std::runtime_error("config: " + where + ": " + what);
}

// yaml-cpp reports a type mismatch with no key context, which makes a typo in a
// long file hard to place. Everything goes through here so the message always
// names the key -- the same shape camera_config.cpp uses, deliberately.
template <typename T>
void read(const YAML::Node& parent, const char* key, const std::string& where, T& out) {
  if (!parent[key]) return;
  try {
    out = parent[key].as<T>();
  } catch (const YAML::Exception&) {
    fail(where + "." + key, "cannot read '" + YAML::Dump(parent[key]) + "' as the expected type");
  }
}

}  // namespace

StereoConfig load_stereo_config(const std::string& path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("config: cannot load " + path + ": " + e.what());
  }

  StereoConfig config;

  if (const YAML::Node stereo = root["stereo"]) {
    // Microseconds in the file, nanoseconds in the code: a tolerance written as
    // 500000 is easy to get wrong by a factor of a thousand, and 500 is not.
    uint64_t tolerance_us = config.tolerance_ns / 1000;
    read(stereo, "tolerance_us", "stereo", tolerance_us);
    config.tolerance_ns = tolerance_us * 1000;

    read(stereo, "queue_frames", "stereo", config.queue_frames);
    read(stereo, "hold_ms", "stereo", config.hold_ms);
    read(stereo, "display", "stereo", config.display);
    read(stereo, "decimate", "stereo", config.decimate);
    read(stereo, "window_width", "stereo", config.window_width);
    read(stereo, "window_height", "stereo", config.window_height);
    read(stereo, "vsync", "stereo", config.vsync);
    read(stereo, "buffer_count", "stereo", config.buffer_count);
  }

  if (const YAML::Node recording = root["recording"]) {
    read(recording, "enabled", "recording", config.record);
    read(recording, "root", "recording", config.record_root);
    read(recording, "staging_frames", "recording", config.staging_frames);
  }

  if (const YAML::Node streams = root["streams"]) {
    if (!streams.IsSequence()) fail("streams", "expected a sequence");
    for (std::size_t i = 0; i < streams.size(); ++i) {
      const std::string where = "streams[" + std::to_string(i) + "]";
      if (!streams[i].IsMap()) fail(where, "expected a map with 'role' and 'serial'");
      StereoStreamConfig stream;
      read(streams[i], "role", where, stream.role);
      read(streams[i], "serial", where, stream.serial);
      if (stream.role.empty()) stream.role = i == 0 ? "left" : "right";
      config.streams.push_back(std::move(stream));
    }
  }

  if (config.streams.empty()) {
    config.streams.push_back({"left", ""});
    config.streams.push_back({"right", ""});
  }
  if (config.streams.size() != 2) {
    fail("streams", "a stereo pair is exactly two streams, found " +
                        std::to_string(config.streams.size()));
  }

  return config;
}

}  // namespace perception
