#include "topic_graph.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace perception {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("topics: " + what); }

const char* residency_name(Residency residency) {
  return residency == Residency::Host ? "host" : "device";
}

std::string pad(const std::string& text, std::size_t width) {
  return text.size() >= width ? text : text + std::string(width - text.size(), ' ');
}

}  // namespace

void TopicGraph::declare(Entry entry) {
  if (entry.info.name.empty() || entry.info.name.front() != '/') {
    fail("'" + entry.info.name + "' is not a topic name -- ROS topics start with '/'");
  }
  if (topics_.find(entry.info.name) != topics_.end()) {
    fail("'" + entry.info.name + "' is declared twice; a name is one producer");
  }
  if (entry.info.residency == Residency::Device && !entry.info.ros_type.empty()) {
    fail("'" + entry.info.name + "' is device-resident but claims the ROS type '" +
         entry.info.ros_type + "'; device buffers have no wire form");
  }
  const std::string name = entry.info.name;
  topics_.emplace(name, std::move(entry));
}

void TopicGraph::declare_image_stream(TopicInfo info, uint32_t stream_index,
                                      std::function<HostFrameRing&()> tap) {
  Entry entry;
  entry.info = std::move(info);
  entry.stream_index = stream_index;
  entry.tap_factory = std::move(tap);
  declare(std::move(entry));
}

void TopicGraph::declare_device_ring(TopicInfo info, uint32_t stream_index,
                                     std::function<DeviceRingBuffer&()> ring) {
  Entry entry;
  entry.info = std::move(info);
  entry.info.residency = Residency::Device;
  entry.stream_index = stream_index;
  entry.ring_factory = std::move(ring);
  declare(std::move(entry));
}

void TopicGraph::declare_host_plane(TopicInfo info, std::function<DownloadStage&()> stage) {
  Entry entry;
  entry.info = std::move(info);
  entry.plane_factory = std::move(stage);
  declare(std::move(entry));
}

bool TopicGraph::has(std::string_view name) const { return topics_.find(name) != topics_.end(); }

const TopicGraph::Entry& TopicGraph::find(std::string_view name) const {
  const auto it = topics_.find(name);
  if (it != topics_.end()) return it->second;

  std::string available;
  for (const auto& [topic, entry] : topics_) available += "\n    " + topic;
  if (available.empty()) available = " (nothing was declared)";
  fail("no topic '" + std::string(name) + "' here. Declared:" + available);
}

TopicGraph::Entry& TopicGraph::find(std::string_view name) {
  return const_cast<Entry&>(static_cast<const TopicGraph*>(this)->find(name));
}

const TopicInfo& TopicGraph::info(std::string_view name) const { return find(name).info; }

uint32_t TopicGraph::stream_index(std::string_view name) const { return find(name).stream_index; }

HostFrameRing& TopicGraph::host_tap(std::string_view name) {
  Entry& entry = find(name);
  if (!entry.tap_factory) {
    fail("'" + std::string(name) + "' is not an image stream, so it has no host tap");
  }
  if (entry.tap == nullptr) entry.tap = &entry.tap_factory();
  return *entry.tap;
}

DeviceRingBuffer& TopicGraph::device_ring(std::string_view name) {
  Entry& entry = find(name);
  if (!entry.ring_factory) {
    fail("'" + std::string(name) + "' is " + residency_name(entry.info.residency) +
         "-resident, so it has no device ring");
  }
  if (entry.ring == nullptr) entry.ring = &entry.ring_factory();
  return *entry.ring;
}

DownloadStage& TopicGraph::download(std::string_view name) {
  Entry& entry = find(name);
  if (!entry.plane_factory) {
    fail("'" + std::string(name) + "' carries no host plane read back from the device");
  }
  if (entry.plane == nullptr) entry.plane = &entry.plane_factory();
  return *entry.plane;
}

std::vector<std::string> TopicGraph::names() const {
  std::vector<std::string> out;
  out.reserve(topics_.size());
  for (const auto& [name, entry] : topics_) out.push_back(name);
  return out;
}

std::string TopicGraph::summary() const {
  if (topics_.empty()) return "topics: none declared\n";

  std::size_t name_width = 0;
  std::size_t type_width = 0;
  for (const auto& [name, entry] : topics_) {
    name_width = std::max(name_width, name.size());
    // "--" stands in for a device topic, which has no wire form to name.
    type_width = std::max(type_width, entry.info.ros_type.empty() ? 2 : entry.info.ros_type.size());
  }

  std::ostringstream out;
  out << "topics:\n";
  for (const auto& [name, entry] : topics_) {
    const TopicInfo& info = entry.info;
    out << "  " << pad(name, name_width) << "  " << pad(residency_name(info.residency), 6) << "  "
        << pad(info.ros_type.empty() ? "--" : info.ros_type, type_width);
    if (!info.frame_id.empty()) out << "  " << info.frame_id;
    if (!info.producer.empty()) out << "  <- " << info.producer;
    out << "\n";
  }
  return out.str();
}

}  // namespace perception
