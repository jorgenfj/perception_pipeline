// The round trip: McapRecorder writes an MCAP, McapReplaySource reads it back
// and feeds a FrameSink. This is the camera-free path the whole project is
// tested and profiled on, so what it has to prove is that a replayed frame is
// the frame that was recorded -- same pixels, same geometry, same order.
//
// It also exercises perception::CdrReader against the writer that produced the
// bytes, which is the half tests/cdr_reader.hpp deliberately does not do.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "heap_frame_sink.hpp"
#include "mcap_recorder.hpp"
#include "mcap_replay_source.hpp"
#include "ros_messages.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

constexpr uint32_t kWidth = 32;
constexpr uint32_t kHeight = 16;
constexpr std::size_t kFrameBytes = static_cast<std::size_t>(kWidth) * kHeight;
constexpr uint32_t kFrames = 4;
constexpr uint64_t kFirstStamp = 1788281225865213416ull;
constexpr uint64_t kPeriodNs = 10'000'000ull;  // 100Hz, so the test is not a wait

// A pattern that is different for every frame and cheap to verify.
std::vector<unsigned char> pattern(uint32_t frame) {
  std::vector<unsigned char> out(kFrameBytes);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<unsigned char>((frame * 37 + i) % 251);
  }
  return out;
}

// Writes kFrames images on /left/image_raw, plus one on /right/image_raw so the
// topic resolution has something to choose between.
std::string write_recording(const std::filesystem::path& root) {
  perception::McapRecorder::Config config;
  config.root = root.string();
  config.compress = false;  // a failure is then a bug here, not in zstd
  config.epoch_offset_ns = 37'000'000'000ll;

  perception::McapRecorder recorder(config);
  const std::string path = recorder.path();

  const auto left = perception::ros_msg::add_topic<perception::ros_msg::ImageMessage>(
      recorder, "/left/image_raw", 100.0);
  const auto right = perception::ros_msg::add_topic<perception::ros_msg::ImageMessage>(
      recorder, "/right/image_raw", 100.0);
  recorder.start();

  perception::ImageDesc desc;
  desc.width = kWidth;
  desc.height = kHeight;
  desc.stride_bytes = kWidth;
  desc.format = perception::PixelFormat::Bayer8_RGGB;

  for (uint32_t i = 0; i < kFrames; ++i) {
    const std::vector<unsigned char> pixels = pattern(i);
    perception::ros_msg::write(
        recorder, left,
        perception::ros_msg::ImageMessage{{kFirstStamp + i * kPeriodNs, "left_optical"},
                                          {desc, pixels.data(), pixels.size()}});
  }
  const std::vector<unsigned char> other = pattern(99);
  perception::ros_msg::write(
      recorder, right,
      perception::ros_msg::ImageMessage{{kFirstStamp, "right_optical"},
                                        {desc, other.data(), other.size()}});

  recorder.close();
  return path;
}

void geometry_comes_from_the_file(const std::string& path) {
  std::printf("the geometry is read out of the recording, not configured\n");

  perception::McapReplaySource::Config config;
  config.path = path;
  config.role = "left";
  config.loop = false;
  perception::McapReplaySource source(config);

  check(source.topic() == "/left/image_raw", "the role resolved to its topic");
  check(source.frame_id() == "left_optical", "and the frame_id came with it");
  check(source.message_count() == kFrames, "the summary counted this topic's messages only");

  const perception::CameraGeometry& g = source.geometry();
  check(g.width == kWidth && g.height == kHeight, "geometry matches what was written");
  check(g.stride_bytes == kWidth, "step became the stride");
  check(g.pixel_format == "BayerRG8", "the ROS encoding mapped back to the camera's name");
  check(g.frame_bytes == kFrameBytes && g.buffer_bytes == kFrameBytes, "and the frame size");
}

void frames_come_back_whole_and_in_order(const std::string& path) {
  std::printf("every frame replays, byte for byte, in the order it was written\n");

  perception::McapReplaySource::Config config;
  config.path = path;
  config.role = "left";
  config.loop = false;
  config.speed = 100.0;  // the pacing is not what this case is about
  perception::McapReplaySource source(config);

  perception::HeapFrameSink sink(4, source.geometry().buffer_bytes);
  source.start(sink);

  std::vector<std::vector<unsigned char>> got;
  std::vector<uint64_t> stamps;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (got.size() < kFrames && std::chrono::steady_clock::now() < deadline) {
    perception::HeapFrameSink::Frame frame;
    if (!sink.pop(frame, std::chrono::milliseconds(50))) continue;
    const auto* p = static_cast<const unsigned char*>(frame.data);
    got.emplace_back(p, p + frame.meta.bytes);
    stamps.push_back(frame.meta.timestamp_ns);
    sink.release(frame.slot);
  }

  source.stop();
  sink.stop();

  check(got.size() == kFrames, "every recorded frame arrived");
  if (got.size() == kFrames) {
    bool bytes_ok = true;
    bool order_ok = true;
    for (uint32_t i = 0; i < kFrames; ++i) {
      if (got[i] != pattern(i)) bytes_ok = false;
      if (i > 0 && stamps[i] <= stamps[i - 1]) order_ok = false;
    }
    check(bytes_ok, "each one byte for byte, and none of them the right camera's");
    check(order_ok, "and their stamps ascend");

    // Rebasing keeps the intervals and moves the origin, so the gaps must be
    // the recorded ones divided by speed -- not the file's original stamps.
    const uint64_t gap = stamps[1] - stamps[0];
    check(gap < kPeriodNs, "rebased onto now, with the intervals scaled by speed");
    check(stamps[0] > kFirstStamp, "so the first stamp is not the file's");
  }
  check(source.finished() && !source.failed(), "played to the end without failing");
  check(source.delivered() == kFrames && source.undecodable() == 0, "counters agree");
}

void verbatim_stamps_are_the_files_own(const std::string& path) {
  std::printf("rebase_timestamps off pushes the recorded stamps through\n");

  perception::McapReplaySource::Config config;
  config.path = path;
  config.role = "left";
  config.loop = false;
  config.speed = 100.0;
  config.rebase_timestamps = false;
  perception::McapReplaySource source(config);

  perception::HeapFrameSink sink(4, source.geometry().buffer_bytes);
  source.start(sink);

  perception::HeapFrameSink::Frame frame;
  const bool popped = sink.pop(frame, std::chrono::seconds(5));
  const uint64_t stamp = popped ? frame.meta.timestamp_ns : 0;
  if (popped) sink.release(frame.slot);

  source.stop();
  sink.stop();

  check(popped && stamp == kFirstStamp, "the first frame carries the stamp that was written");
}

void a_missing_topic_names_what_is_there(const std::string& path) {
  std::printf("asking for a topic the file does not have says what it does have\n");

  perception::McapReplaySource::Config config;
  config.path = path;
  config.role = "centre";

  std::string message;
  try {
    perception::McapReplaySource source(config);
  } catch (const std::exception& e) {
    message = e.what();
  }

  check(!message.empty(), "construction refused it");
  check(message.find("/left/image_raw") != std::string::npos &&
            message.find("/right/image_raw") != std::string::npos,
        "and both image topics are named in the message");
}

}  // namespace

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "perception_mcap_replay_test";
  std::filesystem::remove_all(root);

  int status = 0;
  try {
    const std::string path = write_recording(root);
    geometry_comes_from_the_file(path);
    frames_come_back_whole_and_in_order(path);
    verbatim_stamps_are_the_files_own(path);
    a_missing_topic_names_what_is_there(path);
  } catch (const std::exception& e) {
    std::printf("  [FAIL] threw: %s\n", e.what());
    ++g_failures;
    status = 1;
  }

  std::filesystem::remove_all(root);
  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? status : 1;
}
