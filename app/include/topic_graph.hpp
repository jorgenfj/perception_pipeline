#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace perception {

class DeviceRingBuffer;
class DownloadStage;
class HostFrameRing;

/**
 * @brief Which side of the PCIe bus a topic's buffers live on.
 *
 * Only a host topic can carry a ROS type, and so only a host topic can be
 * recorded or replayed.
 */
enum class Residency : uint8_t { Host, Device };

/**
 * @brief What a topic is, independent of what produces it.
 *
 */
struct TopicInfo {
  std::string name;      ///< "/left/image_raw"
  Residency residency = Residency::Host;
  std::string ros_type;  ///< "sensor_msgs/msg/Image"; empty for a device topic.
  std::string frame_id;

  std::string producer;  ///< "camera 23494258", "mcap /left/image_raw", "ess".
};

/**
 * @brief Where every buffer in the run gets its name.
 */
class TopicGraph {
 public:
  static constexpr uint32_t kNoStream = ~0u;

  /**
   * @brief Declare the raw frames of one pipeline stream.
   * @param stream_index Which stream of the source carries it.
   * @param tap Builds the host fan-out ring, on first resolve. Called at most once.
   */
  void declare_image_stream(TopicInfo info, uint32_t stream_index,
                            std::function<HostFrameRing&()> tap);

  /**
   * @brief Declare a device-resident ring.
   * @param stream_index Which stream of the source it derives from, or
   *        kNoStream for a ring that belongs to no one stream.
   * @param ring Hands back the ring, on first resolve. Called at most once.
   */
  void declare_device_ring(TopicInfo info, uint32_t stream_index,
                           std::function<DeviceRingBuffer&()> ring);

  /**
   * @brief Declare a host plane that a GPU stage reads back.
   * @param stage Builds the readback stage, on first resolve. Called at most once.
   */
  void declare_host_plane(TopicInfo info, std::function<DownloadStage&()> stage);

  /** @brief True if `name` was declared. */
  bool has(std::string_view name) const;

  /**
   * @brief What `name` is.
   * @throws std::runtime_error naming every declared topic. The difference is
   *         usually one character, so the message has to show the alternatives.
   */
  const TopicInfo& info(std::string_view name) const;

  /**
   * @brief Which pipeline stream carries `name`, or kNoStream.
   * @throws std::runtime_error as info() does.
   */
  uint32_t stream_index(std::string_view name) const;

  /**
   * @brief The host fan-out for `name`, building it if this is the first ask.
   * @throws std::runtime_error if `name` is unknown or is not an image stream.
   */
  HostFrameRing& host_tap(std::string_view name);

  /**
   * @brief The device ring behind `name`.
   * @throws std::runtime_error if `name` is unknown or is not device-resident.
   */
  DeviceRingBuffer& device_ring(std::string_view name);

  /**
   * @brief The readback stage behind `name`, building it if this is the first ask.
   * @throws std::runtime_error if `name` is unknown or carries no host plane.
   */
  DownloadStage& download(std::string_view name);

  /** @brief Every declared name, in the order a map gives them: sorted. */
  std::vector<std::string> names() const;

  /**
   * @brief The startup dump: one line per topic, aligned.
   *
   * The whole dataflow in one block, which is the point of naming it at all.
   */
  std::string summary() const;

 private:
  struct Entry {
    TopicInfo info;
    uint32_t stream_index = kNoStream;

    std::function<HostFrameRing&()> tap_factory;
    std::function<DeviceRingBuffer&()> ring_factory;
    std::function<DownloadStage&()> plane_factory;
    
    HostFrameRing* tap = nullptr;
    DeviceRingBuffer* ring = nullptr;
    DownloadStage* plane = nullptr;
  };

  void declare(Entry entry);
  const Entry& find(std::string_view name) const;
  Entry& find(std::string_view name);

  // std::less<> so a string_view looks up without allocating, and sorted so
  // summary() and the error messages come out in a stable order.
  std::map<std::string, Entry, std::less<>> topics_;
};

}  // namespace perception
