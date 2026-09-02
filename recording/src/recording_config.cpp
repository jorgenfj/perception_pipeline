#include "recording_config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <stdexcept>

namespace perception {
namespace {

[[noreturn]] void fail(const std::string& where, const std::string& what) {
  throw std::runtime_error("config: " + where + ": " + what);
}

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

YAML::Node loadFile(const std::string& path) {
  try {
    return YAML::LoadFile(path);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("config: cannot load " + path + ": " + e.what());
  }
}

}  // namespace

RecordingConfig load_recording_config(const std::string& path) {
  const YAML::Node root = loadFile(path);
  RecordingConfig config;

  const YAML::Node recording = root["recording"];
  if (!recording) return config;

  read(recording, "enabled", "recording", config.enabled);
  read(recording, "root", "recording", config.recorder.root);
  read(recording, "compress", "recording", config.recorder.compress);
  read(recording, "buffer_seconds", "recording", config.recorder.buffer_seconds);
  read(recording, "topic_memory_mb", "recording", config.recorder.topic_memory_mb);
  read(recording, "chunk_mb", "recording", config.recorder.chunk_mb);
  read(recording, "flush_seconds", "recording", config.recorder.flush_seconds);

  if (const YAML::Node topics = recording["topics"]) {
    if (!topics.IsSequence()) fail("recording.topics", "expected a list of topic names");
    for (std::size_t i = 0; i < topics.size(); ++i) {
      config.topics.push_back(
          require<std::string>(topics[i], "recording.topics[" + std::to_string(i) + "]"));
    }
  }

  if (!config.enabled) return config;

  if (config.recorder.root.empty()) {
    fail("recording.root", "empty, but recording.enabled is set");
  }
  if (config.recorder.buffer_seconds <= 0.0 || config.recorder.buffer_seconds > 30.0) {
    fail("recording.buffer_seconds", "must be in (0, 30] -- it multiplies every topic's rate");
  }
  if (config.recorder.topic_memory_mb == 0) {
    fail("recording.topic_memory_mb", "must be at least 1");
  }
  if (config.recorder.chunk_mb == 0) {
    fail("recording.chunk_mb", "must be at least 1 -- mcap has no unchunked compressed form");
  }
  if (config.recorder.flush_seconds < 0.0) {
    fail("recording.flush_seconds", "cannot be negative; 0 closes a chunk after every message");
  }
  if (config.topics.empty()) {
    fail("recording.topics", "empty, but recording.enabled is set -- there would be nothing to write");
  }
  for (const std::string& topic : config.topics) {
    if (topic.empty() || topic.front() != '/') {
      fail("recording.topics", "'" + topic + "' is not a topic name -- ROS topics start with '/'");
    }
    if (std::count(config.topics.begin(), config.topics.end(), topic) > 1) {
      fail("recording.topics", "'" + topic + "' is listed twice");
    }
  }

  return config;
}

bool RecordingConfig::records(std::string_view topic) const {
  return std::find(topics.begin(), topics.end(), topic) != topics.end();
}

}  // namespace perception
