// TopicGraph: declaring names, resolving them, and the errors in between.
//
// The laziness cases are the ones that matter most: a run that records nothing
// must still allocate no host tap and no pinned readback pool, and that is only
// true if resolving is what builds them.
//
// No GPU. Every declaration takes a factory, so a device topic can be declared
// and inspected here without a driver -- the factories that would allocate are
// simply never resolved, which is exactly the property being tested.

#include "topic_graph.hpp"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#include "device_ring_buffer.hpp"
#include "download_stage.hpp"
#include "host_frame_ring.hpp"

namespace {

int g_failures = 0;

void check(bool condition, const std::string& what) {
  std::printf("  [%s] %s\n", condition ? "ok" : "FAIL", what.c_str());
  if (!condition) ++g_failures;
}

// Ran the body and returned the message, or "" if nothing was thrown.
template <typename Body>
std::string threw(Body&& body) {
  try {
    body();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

perception::TopicInfo image_info(std::string name, std::string producer) {
  perception::TopicInfo info;
  info.name = std::move(name);
  info.residency = perception::Residency::Host;
  info.ros_type = "sensor_msgs/msg/Image";
  info.frame_id = "left_optical";
  info.producer = std::move(producer);
  return info;
}

void names_resolve_to_what_declared_them() {
  std::printf("names_resolve_to_what_declared_them\n");

  perception::TopicGraph graph;
  perception::HostFrameRing tap(3, 64 * 32, 64, 32);

  graph.declare_image_stream(image_info("/left/image_raw", "camera 23494258"), 0,
                             [&]() -> perception::HostFrameRing& { return tap; });

  check(graph.has("/left/image_raw"), "the declared name is there");
  check(!graph.has("/right/image_raw"), "an undeclared name is not");
  check(graph.info("/left/image_raw").producer == "camera 23494258", "info carries the producer");
  check(graph.info("/left/image_raw").ros_type == "sensor_msgs/msg/Image", "and the ROS type");
  check(graph.stream_index("/left/image_raw") == 0, "and which stream carries it");
  check(&graph.host_tap("/left/image_raw") == &tap, "host_tap resolves to the real ring");
  check(graph.names().size() == 1, "names() lists it");
}

void an_unknown_name_lists_the_known_ones() {
  std::printf("an_unknown_name_lists_the_known_ones\n");

  perception::TopicGraph graph;
  perception::HostFrameRing tap(3, 64 * 32, 64, 32);
  graph.declare_image_stream(image_info("/left/image_raw", "camera"), 0,
                             [&]() -> perception::HostFrameRing& { return tap; });
  graph.declare_image_stream(image_info("/right/image_raw", "camera"), 1,
                             [&]() -> perception::HostFrameRing& { return tap; });

  const std::string message = threw([&] { graph.info("/centre/image_raw"); });
  check(!message.empty(), "resolving an unknown name throws");
  check(message.find("/centre/image_raw") != std::string::npos, "the message names what was asked");
  check(message.find("/left/image_raw") != std::string::npos, "and lists the first alternative");
  check(message.find("/right/image_raw") != std::string::npos, "and the second");
}

void a_name_is_one_producer() {
  std::printf("a_name_is_one_producer\n");

  perception::TopicGraph graph;
  perception::HostFrameRing tap(3, 64 * 32, 64, 32);
  graph.declare_image_stream(image_info("/left/image_raw", "camera"), 0,
                             [&]() -> perception::HostFrameRing& { return tap; });

  const std::string twice = threw([&] {
    graph.declare_image_stream(image_info("/left/image_raw", "mcap"), 1,
                               [&]() -> perception::HostFrameRing& { return tap; });
  });
  check(twice.find("declared twice") != std::string::npos, "declaring a name twice throws");

  perception::TopicInfo bare = image_info("left/image_raw", "camera");
  const std::string unrooted = threw([&] {
    graph.declare_image_stream(bare, 2, [&]() -> perception::HostFrameRing& { return tap; });
  });
  check(unrooted.find("start with '/'") != std::string::npos, "a name without a leading / throws");
}

void a_factory_runs_once_and_only_when_asked() {
  std::printf("a_factory_runs_once_and_only_when_asked\n");

  perception::TopicGraph graph;
  std::unique_ptr<perception::HostFrameRing> tap;
  int built = 0;

  graph.declare_image_stream(image_info("/left/image_raw", "camera"), 0,
                             [&]() -> perception::HostFrameRing& {
                               ++built;
                               tap = std::make_unique<perception::HostFrameRing>(3, 64 * 32, 64, 32);
                               return *tap;
                             });

  check(built == 0, "declaring builds nothing");
  check(graph.info("/left/image_raw").ros_type == "sensor_msgs/msg/Image",
        "and info() still answers without building");

  perception::HostFrameRing& first = graph.host_tap("/left/image_raw");
  check(built == 1, "the first resolve builds it");

  perception::HostFrameRing& second = graph.host_tap("/left/image_raw");
  check(built == 1, "a second resolve does not build it again");
  check(&first == &second, "and hands back the same ring");
}

// A device ring would need a driver to construct, so its factory is one that
// fails the test if anything resolves it. Nothing here should.
perception::DeviceRingBuffer& unreachable_ring() {
  throw std::runtime_error("the device ring factory was resolved, and should not have been");
}

void a_device_topic_is_not_recordable() {
  std::printf("a_device_topic_is_not_recordable\n");

  perception::TopicGraph graph;

  perception::TopicInfo info;
  info.name = "/left/image_color";
  info.frame_id = "left_optical";
  info.producer = "debayer";
  graph.declare_device_ring(info, 0, unreachable_ring);

  check(graph.info("/left/image_color").residency == perception::Residency::Device,
        "it declares as device-resident");
  check(graph.info("/left/image_color").ros_type.empty(), "and carries no ROS type");

  const std::string as_tap = threw([&] { graph.host_tap("/left/image_color"); });
  check(as_tap.find("not an image stream") != std::string::npos,
        "asking for its host tap says why there is none");

  const std::string as_plane = threw([&] { graph.download("/left/image_color"); });
  check(as_plane.find("no host plane") != std::string::npos,
        "and asking for a host plane says the same");

  perception::TopicInfo typed = info;
  typed.name = "/left/image_typed";
  typed.ros_type = "sensor_msgs/msg/Image";
  const std::string claimed = threw([&] { graph.declare_device_ring(typed, 0, unreachable_ring); });
  check(claimed.find("no wire form") != std::string::npos,
        "a device topic claiming a ROS type is refused");
}

void a_host_topic_has_no_device_ring() {
  std::printf("a_host_topic_has_no_device_ring\n");

  perception::TopicGraph graph;
  perception::HostFrameRing tap(3, 64 * 32, 64, 32);
  graph.declare_image_stream(image_info("/left/image_raw", "camera"), 0,
                             [&]() -> perception::HostFrameRing& { return tap; });

  const std::string message = threw([&] { graph.device_ring("/left/image_raw"); });
  check(message.find("host-resident") != std::string::npos,
        "asking a host topic for a device ring says which side it is on");
}

void the_summary_names_every_topic() {
  std::printf("the_summary_names_every_topic\n");

  perception::TopicGraph graph;
  perception::HostFrameRing tap(3, 64 * 32, 64, 32);

  graph.declare_image_stream(image_info("/left/image_raw", "mcap /left/image_raw"), 0,
                             [&]() -> perception::HostFrameRing& { return tap; });

  perception::TopicInfo colour;
  colour.name = "/left/image_color";
  colour.frame_id = "left_optical";
  colour.producer = "debayer";
  graph.declare_device_ring(colour, 0, unreachable_ring);

  const std::string summary = graph.summary();
  std::printf("%s", summary.c_str());
  check(summary.find("/left/image_raw") != std::string::npos, "the host topic is listed");
  check(summary.find("/left/image_color") != std::string::npos, "and the device one");
  check(summary.find("sensor_msgs/msg/Image") != std::string::npos, "with its ROS type");
  check(summary.find("mcap /left/image_raw") != std::string::npos, "and its producer");
  check(summary.find("left_optical") != std::string::npos, "and the frame it is measured in");
}

}  // namespace

int main() {
  names_resolve_to_what_declared_them();
  an_unknown_name_lists_the_known_ones();
  a_name_is_one_producer();
  a_factory_runs_once_and_only_when_asked();
  a_device_topic_is_not_recordable();
  a_host_topic_has_no_device_ring();
  the_summary_names_every_topic();

  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
