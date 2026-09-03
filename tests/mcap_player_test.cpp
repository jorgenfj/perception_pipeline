// The round trip: McapRecorder writes an MCAP, McapPlayer reads it back and
// feeds whatever subscribed. This is the camera-free path the whole project is
// tested and profiled on, so what it has to prove is that a replayed message is
// the message that was recorded -- same bytes, same geometry, same order -- and
// that a stereo pair comes back with the skew it was recorded with.
//
// It also exercises perception::CdrReader against the writer that produced the
// bytes, which is the half tests/cdr_reader.hpp deliberately does not do.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "heap_frame_sink.hpp"
#include "image_replay_source.hpp"
#include "mcap_player.hpp"
#include "mcap_recorder.hpp"
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

// The right eye trails the left by this much, every frame. Small enough to be a
// real rig's skew and large enough that losing it would be obvious.
constexpr uint64_t kSkewNs = 250'000ull;

// Two IMU samples per frame period, so the interleaving has something to say.
constexpr uint32_t kImuPerFrame = 2;

constexpr const char* kOddTopic = "/vendor/blob";
constexpr const char* kOddType = "vendor_msgs/msg/Blob";

// A pattern that is different for every frame and cheap to verify.
std::vector<unsigned char> pattern(uint32_t frame) {
  std::vector<unsigned char> out(kFrameBytes);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<unsigned char>((frame * 37 + i) % 251);
  }
  return out;
}

perception::ros_msg::Imu imu_sample(uint32_t i) {
  perception::ros_msg::Imu imu;
  imu.angular_velocity = {0.1 * i, 0.2 * i, 0.3 * i};
  imu.linear_acceleration = {1.0 * i, 2.0 * i, 9.81};
  for (std::size_t k = 0; k < 9; ++k) {
    imu.angular_velocity_covariance[k] = 0.001 * (k + i);
    imu.linear_acceleration_covariance[k] = 0.002 * (k + i);
    imu.orientation_covariance[k] = 0.003 * (k + i);
  }
  imu.orientation = {0.0, 0.0, 0.0, 1.0};
  return imu;
}

// A stereo pair with a deliberate skew, an IMU stream, and one topic carrying a
// schema nothing in this tree has ever heard of.
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
  const auto imu = perception::ros_msg::add_topic<perception::ros_msg::ImuMessage>(
      recorder, "/imu/data", 200.0);

  // Declared the raw way, bypassing ros_messages entirely: the player must not
  // need to know a type to replay it.
  perception::McapRecorder::Topic odd;
  odd.name = kOddTopic;
  odd.type = kOddType;
  odd.schema = "uint64 value\n";
  odd.rate_hz = 100.0;
  odd.max_message_bytes = 64;
  const perception::McapRecorder::TopicId blob = recorder.add_topic(odd);

  recorder.start();

  perception::ImageDesc desc;
  desc.width = kWidth;
  desc.height = kHeight;
  desc.stride_bytes = kWidth;
  desc.format = perception::PixelFormat::Bayer8_RGGB;

  for (uint32_t i = 0; i < kFrames; ++i) {
    const uint64_t stamp = kFirstStamp + i * kPeriodNs;
    const std::vector<unsigned char> pixels = pattern(i);
    perception::ros_msg::write(
        recorder, left,
        perception::ros_msg::ImageMessage{{stamp, "left_optical"},
                                          {desc, pixels.data(), pixels.size()}});

    const std::vector<unsigned char> other = pattern(i + 100);
    perception::ros_msg::write(
        recorder, right,
        perception::ros_msg::ImageMessage{{stamp + kSkewNs, "right_optical"},
                                          {desc, other.data(), other.size()}});

    for (uint32_t k = 0; k < kImuPerFrame; ++k) {
      const uint32_t index = i * kImuPerFrame + k;
      perception::ros_msg::write(
          recorder, imu,
          perception::ros_msg::ImuMessage{{stamp + k * (kPeriodNs / kImuPerFrame), "imu_link"},
                                          imu_sample(index)});
    }

    recorder.push(blob, stamp, [i](perception::CdrWriter& cdr) { cdr.u64(i); });
  }

  recorder.close();
  return path;
}

perception::McapPlayer::Config base_config(const std::string& path) {
  perception::McapPlayer::Config config;
  config.path = path;
  config.loop = false;
  config.speed = 100.0;  // the pacing is not what most of these cases are about
  return config;
}

// Drains a sink until `want` frames arrive or the deadline passes.
struct Drained {
  std::vector<std::vector<unsigned char>> pixels;
  std::vector<uint64_t> stamps;
};

Drained drain(perception::HeapFrameSink& sink, std::size_t want) {
  Drained out;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (out.pixels.size() < want && std::chrono::steady_clock::now() < deadline) {
    perception::HeapFrameSink::Frame frame;
    if (!sink.pop(frame, std::chrono::milliseconds(50))) continue;
    const auto* p = static_cast<const unsigned char*>(frame.data);
    out.pixels.emplace_back(p, p + frame.meta.bytes);
    out.stamps.push_back(frame.meta.timestamp_ns);
    sink.release(frame.slot);
  }
  return out;
}

// The last IMU sample of the file sits after the last frame, so draining the
// image sinks is not the same as the pass being over.
bool wait_finished(const perception::McapPlayer& player) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!player.finished() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return player.finished();
}

void geometry_comes_from_the_file(const std::string& path) {
  std::printf("the geometry is read out of the recording, not configured\n");

  perception::McapPlayer player(base_config(path));
  perception::ImageReplaySource source(player, "/left/image_raw");

  check(source.topic() == "/left/image_raw", "the source is bound to its topic");
  check(source.frame_id() == "left_optical", "and the frame_id came with it");
  check(source.message_count() == kFrames, "the summary counted this topic's messages only");

  const perception::CameraGeometry& g = source.geometry();
  check(g.width == kWidth && g.height == kHeight, "geometry matches what was written");
  check(g.stride_bytes == kWidth, "step became the stride");
  check(g.pixel_format == "BayerRG8", "the ROS encoding mapped back to the camera's name");
  check(g.frame_bytes == kFrameBytes && g.buffer_bytes == kFrameBytes, "and the frame size");

  check(player.image_topics().size() == 2, "both image topics resolved");
  check(player.image_topics()[0] == "/left/image_raw", "and left sorted to stream 0");
  check(player.recorded_clock().find("UTC") != std::string::npos,
        "the recorder's clock note came back as provenance");
}

void every_topic_replays_by_default(const std::string& path) {
  std::printf("an empty topics list replays the whole file\n");

  perception::McapPlayer player(base_config(path));
  check(player.channels().size() == 4, "all four channels were selected");

  std::vector<std::string> seen_order;
  uint32_t imu_seen = 0;
  uint32_t blob_seen = 0;
  bool imu_decoded = true;
  std::string blob_schema;

  player.subscribe("/imu/data", [&](const perception::ReplayMessage& message) {
    perception::ros_msg::ImuMessage back;
    if (!perception::ros_msg::decode(message, back)) {
      imu_decoded = false;
      return;
    }
    const perception::ros_msg::Imu expected = imu_sample(imu_seen);
    if (back.data.angular_velocity != expected.angular_velocity ||
        back.data.linear_acceleration != expected.linear_acceleration ||
        back.data.linear_acceleration_covariance != expected.linear_acceleration_covariance ||
        back.data.orientation_covariance != expected.orientation_covariance) {
      imu_decoded = false;
    }
    if (back.header.frame_id != "imu_link") imu_decoded = false;
    ++imu_seen;
    seen_order.push_back("/imu/data");
  });

  // No decoder, and none needed: the bytes and the schema name are enough to
  // hand a message on to something this build has never heard of.
  player.subscribe(kOddTopic, [&](const perception::ReplayMessage& message) {
    blob_schema = std::string(message.schema_name);
    if (message.size >= 12) ++blob_seen;
    seen_order.push_back(kOddTopic);
  });

  perception::ImageReplaySource left(player, "/left/image_raw");
  perception::ImageReplaySource right(player, "/right/image_raw");
  perception::HeapFrameSink left_sink(4, left.geometry().buffer_bytes);
  perception::HeapFrameSink right_sink(4, right.geometry().buffer_bytes);

  left.start(left_sink);
  right.start(right_sink);

  const Drained got_left = drain(left_sink, kFrames);
  const Drained got_right = drain(right_sink, kFrames);
  const bool done = wait_finished(player);

  player.stop();
  left_sink.stop();
  right_sink.stop();

  check(done, "the pass ran to the end of the file");
  check(got_left.pixels.size() == kFrames, "every left frame arrived");
  check(got_right.pixels.size() == kFrames, "every right frame arrived");
  check(imu_seen == kFrames * kImuPerFrame, "and every IMU sample");
  check(blob_seen == kFrames, "and every message of the type nothing here knows");
  check(blob_schema == kOddType, "which carried its schema name through untouched");
  check(imu_decoded, "the IMU samples decode field for field, covariances included");

  if (got_left.pixels.size() == kFrames && got_right.pixels.size() == kFrames) {
    bool bytes_ok = true;
    bool order_ok = true;
    for (uint32_t i = 0; i < kFrames; ++i) {
      if (got_left.pixels[i] != pattern(i)) bytes_ok = false;
      if (got_right.pixels[i] != pattern(i + 100)) bytes_ok = false;
      if (i > 0 && got_left.stamps[i] <= got_left.stamps[i - 1]) order_ok = false;
    }
    check(bytes_ok, "each one byte for byte, and neither eye carrying the other's");
    check(order_ok, "and their stamps ascend");

    // The whole reason one player drives both eyes: the recorded skew is a
    // property of the pair, and two independently paced replays would lose it.
    bool skew_ok = true;
    for (uint32_t i = 0; i < kFrames; ++i) {
      const int64_t skew =
          static_cast<int64_t>(got_right.stamps[i]) - static_cast<int64_t>(got_left.stamps[i]);
      // Rebasing divides every interval by speed, the skew included.
      if (skew != static_cast<int64_t>(kSkewNs / 100)) skew_ok = false;
    }
    check(skew_ok, "and the recorded stereo skew survived, scaled by speed and nothing else");

    const uint64_t gap = got_left.stamps[1] - got_left.stamps[0];
    check(gap < kPeriodNs, "rebased onto now, with the intervals scaled by speed");
    check(got_left.stamps[0] > kFirstStamp, "so the first stamp is not the file's");
  }

  check(left.finished() && !left.failed(), "played to the end without failing");
  check(left.delivered() == kFrames && left.undecodable() == 0, "counters agree");
}

void filters_select_what_plays(const std::string& path) {
  std::printf("topics and exclude choose what comes off the file\n");

  {
    perception::McapPlayer::Config config = base_config(path);
    config.topics = {"/imu/data", "/left/image_raw"};
    perception::McapPlayer player(config);
    check(player.channels().size() == 2, "an allowlist takes only what it names");
    check(player.image_topics().size() == 1 && player.image_topics()[0] == "/left/image_raw",
          "and narrowing it to one eye replays as a mono run");
  }
  {
    perception::McapPlayer::Config config = base_config(path);
    config.exclude = {kOddTopic};
    perception::McapPlayer player(config);
    bool has_odd = false;
    for (const auto& channel : player.channels()) {
      if (channel.topic == kOddTopic) has_odd = true;
    }
    check(player.channels().size() == 3 && !has_odd, "a denylist drops what it names");
  }
  {
    // An explicit list is what the replay is for, so it survives a filter that
    // would otherwise have dropped it.
    perception::McapPlayer::Config config = base_config(path);
    config.topics = {"/imu/data"};
    config.image_topics = {"/left/image_raw", "/right/image_raw"};
    perception::McapPlayer player(config);
    check(player.channels().size() == 3, "an explicit image list comes back regardless");
  }
  {
    perception::McapPlayer::Config config = base_config(path);
    config.image_topics = {"/left/image_raw"};
    config.exclude = {"/left/image_raw"};
    std::string message;
    try {
      perception::McapPlayer player(config);
    } catch (const std::exception& e) {
      message = e.what();
    }
    check(message.find("cannot be both") != std::string::npos,
          "asking for a topic to be both fed and dropped refuses rather than half-obeying");
  }
}

void a_missing_topic_names_what_is_there(const std::string& path) {
  std::printf("asking for a topic the file does not have says what it does have\n");

  perception::McapPlayer::Config config = base_config(path);
  config.image_topics = {"/centre/image_raw"};

  std::string message;
  try {
    perception::McapPlayer player(config);
  } catch (const std::exception& e) {
    message = e.what();
  }

  check(!message.empty(), "construction refused it");
  check(message.find("/left/image_raw") != std::string::npos &&
            message.find("/right/image_raw") != std::string::npos,
        "and both image topics are named in the message");
}

void verbatim_stamps_are_the_files_own(const std::string& path) {
  std::printf("rebase_timestamps off pushes the recorded stamps through\n");

  perception::McapPlayer::Config config = base_config(path);
  config.rebase_timestamps = false;
  perception::McapPlayer player(config);

  perception::ImageReplaySource left(player, "/left/image_raw");
  perception::ImageReplaySource right(player, "/right/image_raw");
  perception::HeapFrameSink left_sink(4, left.geometry().buffer_bytes);
  perception::HeapFrameSink right_sink(4, right.geometry().buffer_bytes);
  left.start(left_sink);
  right.start(right_sink);

  const Drained got_left = drain(left_sink, 1);
  const Drained got_right = drain(right_sink, 1);

  player.stop();
  left_sink.stop();
  right_sink.stop();

  check(!got_left.stamps.empty() && got_left.stamps[0] == kFirstStamp,
        "the first frame carries the stamp that was written");
  check(!got_right.stamps.empty() && got_right.stamps[0] == kFirstStamp + kSkewNs,
        "and the other eye carries its own, skew and all");
}

void the_window_clips_the_range(const std::string& path) {
  std::printf("start_seconds and duration_seconds clip what plays\n");

  perception::McapPlayer::Config config = base_config(path);
  // Frames sit at 0, 1, 2 and 3 periods. Starting at 1.5 drops the first two;
  // running for 2.2 from there reaches the fourth's instant but not past it, so
  // the window is frames 2 and 3 and the ends are tested from both sides.
  config.start_seconds = static_cast<double>(kPeriodNs) / 1e9 * 1.5;
  config.duration_seconds = static_cast<double>(kPeriodNs) / 1e9 * 2.2;

  perception::McapPlayer player(config);
  perception::ImageReplaySource left(player, "/left/image_raw");
  perception::HeapFrameSink sink(4, left.geometry().buffer_bytes);
  left.start(sink);

  const Drained got = drain(sink, 2);
  player.stop();
  sink.stop();

  check(got.pixels.size() == 2 && got.pixels[0] == pattern(2) && got.pixels[1] == pattern(3),
        "the window starts at the first frame inside it and ends where it says");
  check(left.delivered() == 2, "and nothing outside it was replayed");
}

}  // namespace

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "perception_mcap_player_test";
  std::filesystem::remove_all(root);

  int status = 0;
  try {
    const std::string path = write_recording(root);
    geometry_comes_from_the_file(path);
    every_topic_replays_by_default(path);
    filters_select_what_plays(path);
    a_missing_topic_names_what_is_there(path);
    verbatim_stamps_are_the_files_own(path);
    the_window_clips_the_range(path);
  } catch (const std::exception& e) {
    std::printf("  [FAIL] threw: %s\n", e.what());
    ++g_failures;
    status = 1;
  }

  std::filesystem::remove_all(root);
  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? status : 1;
}
