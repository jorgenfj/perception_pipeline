#include "replay_config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

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

std::vector<std::string> readTopics(const YAML::Node& parent, const char* key,
                                    const std::string& where) {
  std::vector<std::string> out;
  const YAML::Node node = parent[key];
  if (!node) return out;
  const std::string label = where + "." + key;
  if (!node.IsSequence()) fail(label, "expected a list of topic names");

  for (std::size_t i = 0; i < node.size(); ++i) {
    std::string topic = require<std::string>(node[i], label + "[" + std::to_string(i) + "]");
    if (topic.empty() || topic.front() != '/') {
      fail(label, "'" + topic + "' is not a topic name -- ROS topics start with '/'");
    }
    if (std::find(out.begin(), out.end(), topic) != out.end()) {
      fail(label, "'" + topic + "' is listed twice");
    }
    out.push_back(std::move(topic));
  }
  return out;
}

}  // namespace

McapPlayer::Config load_replay_config(const std::string& path) {
  const YAML::Node root = loadFile(path);
  McapPlayer::Config config;

  const YAML::Node source = root["source"];
  if (!source) return config;

  // The section used to name one eye with `role:` / `topic:`. Silently ignoring
  // either would replay the wrong camera into a stereo pipeline that cannot
  // tell, so they are an error rather than a no-op.
  for (const char* gone : {"role", "topic"}) {
    if (source[gone]) {
      fail(std::string("source.") + gone,
           "no longer exists -- list the image topics that feed the pipeline under "
           "source.image_topics, in stream order");
    }
  }

  read(source, "recording", "source", config.path);
  read(source, "speed", "source", config.speed);
  read(source, "loop", "source", config.loop);
  read(source, "rebase_timestamps", "source", config.rebase_timestamps);
  read(source, "slot_wait_ms", "source", config.slot_wait_ms);
  read(source, "start_seconds", "source", config.start_seconds);
  read(source, "duration_seconds", "source", config.duration_seconds);

  config.image_topics = readTopics(source, "image_topics", "source");
  config.topics = readTopics(source, "topics", "source");
  config.exclude = readTopics(source, "exclude", "source");

  if (config.speed <= 0.0) fail("source.speed", "must be positive");
  if (config.start_seconds < 0.0) fail("source.start_seconds", "cannot be negative");
  if (config.duration_seconds < 0.0) {
    fail("source.duration_seconds", "cannot be negative; 0 runs to the end of the file");
  }

  return config;
}

}  // namespace perception
