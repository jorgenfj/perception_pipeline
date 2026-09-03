// Round trip for the MCAP recorder. No GPU, no camera, no ROS: the frames are
// synthetic and the file is read back with the same vendored library that wrote
// it, then decoded by hand.
//
// Decoding by hand is the point. Writing CDR that mcap accepts proves nothing --
// mcap stores opaque bytes and would happily store garbage. What has to be true
// is that the field alignment is right, because a CDR stream with one bad
// alignment still parses, into plausible-looking nonsense, and the first thing
// that would notice is ros2 bag on a different machine weeks later.

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <array>
#include <sstream>
#include <type_traits>
#include <string>
#include <vector>

#include <mcap/mcap.hpp>

#include "cdr_reader.hpp"
#include "mcap_recorder.hpp"
#include "ros_messages.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

// Bayer, so the encoding the reader sees has to come from the format rather
// than from a string somebody typed.
const perception::ImageDesc kImageDesc =
    perception::packed_desc(8, 4, perception::PixelFormat::Bayer8_RGGB);

constexpr uint32_t kImageW = 8;
constexpr uint32_t kImageH = 4;
constexpr uint32_t kDispW = 6;
constexpr uint32_t kDispH = 3;

std::vector<unsigned char> make_image(uint8_t seed) {
  std::vector<unsigned char> out(static_cast<std::size_t>(kImageW) * kImageH);
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<unsigned char>(seed + i);
  return out;
}

// The property the per-topic budget exists for: on the shared queue this
// replaced, a flood on one topic evicted every message of every other.
void topics_do_not_evict_each_other(const std::filesystem::path& root) {
  std::printf("a flooded topic does not evict a quiet one\n");

  perception::McapRecorder::Config config;
  config.root = root.string();
  config.compress = false;

  perception::McapRecorder recorder(config);

  perception::McapRecorder::Topic flood_declared;
  flood_declared.name = "/flood";
  flood_declared.type = "test_msgs/msg/Tiny";
  flood_declared.schema = "uint64 value\n";
  flood_declared.max_message_bytes = 64;
  flood_declared.queue_depth = 2;  // deliberately tiny, so the flood must overflow it
  const perception::McapRecorder::TopicId flood = recorder.add_topic(flood_declared);

  perception::McapRecorder::Topic quiet_declared = flood_declared;
  quiet_declared.name = "/quiet";
  quiet_declared.queue_depth = 16;
  const perception::McapRecorder::TopicId quiet = recorder.add_topic(quiet_declared);

  recorder.start();

  constexpr int kFlood = 100000;
  constexpr int kQuiet = 10;
  for (int i = 0; i < kFlood; ++i) {
    recorder.push(flood, 1000 + i, [i](perception::CdrWriter& cdr) {
      cdr.u64(static_cast<uint64_t>(i));
    });
    // Interleaved, so the quiet topic is pushing throughout the flood and not
    // before or after it.
    if (i % (kFlood / kQuiet) == 0) {
      recorder.push(quiet, 1000 + i, [i](perception::CdrWriter& cdr) {
        cdr.u64(static_cast<uint64_t>(i));
      });
    }
  }
  recorder.close();

  // Deterministic: the budget is 16 and only 10 are pushed, so no scheduling
  // order can drop one. This is the assertion the shared queue would fail.
  check(recorder.drops(quiet) == 0, "the quiet topic dropped nothing");
  check(recorder.written(quiet) == kQuiet, "and wrote every message it pushed");

  // Nothing vanishes silently: every push is either written or counted.
  check(recorder.written(flood) + recorder.drops(flood) == kFlood,
        "every flooded message was either written or counted as a drop");
  check(recorder.grew() == 0, "no encode outgrew its buffer");

  // Not a check: a machine fast enough to drain a depth-2 queue between pushes
  // would make this flaky, and it proves nothing either way.
  std::printf("       (flood wrote %llu, dropped %llu)\n",
              static_cast<unsigned long long>(recorder.written(flood)),
              static_cast<unsigned long long>(recorder.drops(flood)));

  std::filesystem::remove(recorder.path());
}

// Two topics of one type share a schema record, and a type declared twice with
// different text is refused: a file carrying two definitions of one type is one
// nobody can trust.
void schema_sharing_and_lifecycle(const std::filesystem::path& root) {
  std::printf("topic declaration rules\n");

  perception::McapRecorder::Config config;
  config.root = root.string();
  config.compress = false;

  perception::McapRecorder recorder(config);

  perception::McapRecorder::Topic declared;
  declared.name = "/a";
  declared.type = "test_msgs/msg/Tiny";
  declared.schema = "uint64 value\n";
  declared.max_message_bytes = 64;
  declared.rate_hz = 1.0;
  recorder.add_topic(declared);

  declared.name = "/b";
  recorder.add_topic(declared);

  bool refused = false;
  try {
    perception::McapRecorder::Topic conflicting = declared;
    conflicting.name = "/c";
    conflicting.schema = "uint64 something_else\n";
    recorder.add_topic(conflicting);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  check(refused, "one type declared twice with different text is refused");

  bool no_budget_refused = false;
  try {
    perception::McapRecorder::Topic sizeless = declared;
    sizeless.name = "/d";
    sizeless.max_message_bytes = 0;
    recorder.add_topic(sizeless);
  } catch (const std::runtime_error&) {
    no_budget_refused = true;
  }
  check(no_budget_refused, "a topic declaring no max_message_bytes is refused");

  recorder.start();

  bool after_start_refused = false;
  try {
    declared.name = "/late";
    recorder.add_topic(declared);
  } catch (const std::runtime_error&) {
    after_start_refused = true;
  }
  check(after_start_refused, "add_topic() after start() is refused");

  const std::string path = recorder.path();
  recorder.close();

  mcap::McapReader reader;
  if (reader.open(path).ok()) {
    reader.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);
    check(reader.channels().size() == 2, "two channels");
    check(reader.schemas().size() == 1, "sharing one schema record between them");
    reader.close();
  }
  std::filesystem::remove(path);
}

// close() must not throw away what is still queued: a recorder that lost its
// tail would lose the end of every run, which is usually the interesting part.
void the_tail_is_not_lost(const std::filesystem::path& root) {
  std::printf("close() drains what is queued\n");

  perception::McapRecorder::Config config;
  config.root = root.string();
  config.compress = false;

  perception::McapRecorder recorder(config);

  perception::McapRecorder::Topic declared;
  declared.name = "/burst";
  declared.type = "test_msgs/msg/Tiny";
  declared.schema = "uint64 value\n";
  declared.max_message_bytes = 64;
  declared.queue_depth = 64;
  const perception::McapRecorder::TopicId topic = recorder.add_topic(declared);
  recorder.start();

  uint64_t pushed = 0;
  for (int i = 0; i < 64; ++i) {
    if (recorder.push(topic, 1000 + i, [i](perception::CdrWriter& cdr) {
          cdr.u64(static_cast<uint64_t>(i));
        })) {
      ++pushed;
    }
  }
  const std::string path = recorder.path();
  recorder.close();

  check(recorder.written(topic) == pushed, "every accepted message survived close()");
  check(recorder.drops(topic) == 0, "and none was dropped on the way out");
  std::filesystem::remove(path);
}

// Every type a schema references has to be defined in the same schema text.
// The generator closes this walk; this is what says so, and it is the check
// that a hand-edited ros_schemas.hpp would fail.
void schemas_are_self_contained(const std::string& path) {
  std::printf("every schema defines every type it references\n");

  mcap::McapReader reader;
  if (!reader.open(path).ok()) {
    check(false, "reopening the file for the schema check");
    return;
  }
  reader.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);

  for (const auto& [id, schema] : reader.schemas()) {
    const std::string text(reinterpret_cast<const char*>(schema->data.data()),
                           schema->data.size());

    bool closed = true;
    std::string missing;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
      const std::size_t hash = line.find('#');
      if (hash != std::string::npos) line.resize(hash);
      if (line.rfind("MSG:", 0) == 0 || line.rfind("===", 0) == 0) continue;

      std::istringstream tokens(line);
      std::string type;
      if (!(tokens >> type)) continue;
      const std::size_t bracket = type.find('[');
      if (bracket != std::string::npos) type.resize(bracket);
      if (type.find('/') == std::string::npos) continue;  // a primitive

      if (text.find("MSG: " + type) == std::string::npos) {
        closed = false;
        missing = type;
        break;
      }
    }
    check(closed, schema->name + " defines every type it references" +
                      (closed ? "" : " (missing " + missing + ")"));
  }
  reader.close();
}

// Every sensor type, written and decoded field by field.
//
// The covariances here are deliberately ASYMMETRIC. Eigen stores column-major
// and ROS writes covariances row-major, so a transposed encode is a real and
// easy bug -- and a symmetric matrix, which most covariances are, agrees with
// it perfectly. These do not.
void sensor_messages_round_trip(const std::filesystem::path& root) {
  std::printf("sensor messages round trip\n");

  // Compile-time half of the checklist: a type missing its traits or its
  // encoder fails here, by name, rather than deep in a template.
  static_assert(perception::ros_msg::Recordable<perception::ros_msg::ImuMessage>);
  static_assert(perception::ros_msg::Recordable<perception::ros_msg::MagneticFieldMessage>);
  static_assert(perception::ros_msg::Recordable<perception::ros_msg::FluidPressureMessage>);

  // The point of splitting the header off: this is what a CAN or UART thread
  // accumulates into, and it can cross a lock-free ring as bytes.
  static_assert(std::is_trivially_copyable_v<perception::ros_msg::Imu>);

  const uint64_t stamp = 1788281225865213416ull;

  // Asymmetric and every element distinct -- covariance[r*N + c] == 10^k*r + c
  // -- so an encoder that reordered or reversed the run could not pass.
  auto asymmetric3 = [](double scale) {
    std::array<double, 9> m{};
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) m[r * 3 + c] = (10.0 * r + c) * scale;
    }
    return m;
  };

  std::string path;
  {
    perception::McapRecorder::Config config;
    config.root = root.string();
    config.compress = false;

    perception::McapRecorder recorder(config);

    // No context: these carry their own header, so a topic is a name and a rate.
    const auto imu = perception::ros_msg::add_topic<perception::ros_msg::ImuMessage>(
        recorder, "/imu/data", 400.0);
    const auto mag = perception::ros_msg::add_topic<perception::ros_msg::MagneticFieldMessage>(
        recorder, "/imu/mag", 20.0);
    const auto pressure =
        perception::ros_msg::add_topic<perception::ros_msg::FluidPressureMessage>(
            recorder, "/pressure", 20.0);
    recorder.start();
    path = recorder.path();

    perception::ros_msg::ImuMessage imu_sample;
    // "imu_link" is 8 characters, so the string writes 4 + 9 = 13 bytes and the
    // float64 after it genuinely needs padding.
    imu_sample.header = {stamp, "imu_link"};
    imu_sample.data.orientation = {0.5, -0.5, 0.5, 0.5};  // x, y, z, w
    imu_sample.data.angular_velocity = {0.125, -0.25, 0.375};
    imu_sample.data.linear_acceleration = {-1.5, 9.75, 0.0625};
    imu_sample.data.orientation_covariance = asymmetric3(1.0);
    imu_sample.data.angular_velocity_covariance = asymmetric3(2.0);
    imu_sample.data.linear_acceleration_covariance = asymmetric3(3.0);
    check(perception::ros_msg::write(recorder, imu, imu_sample), "imu accepted");

    perception::ros_msg::MagneticFieldMessage mag_sample;
    mag_sample.header = {stamp + 1, "dvl_link"};
    mag_sample.data.magnetic_field = {1.5e-5, -2.5e-5, 4.75e-5};
    mag_sample.data.magnetic_field_covariance = asymmetric3(1.0);
    check(perception::ros_msg::write(recorder, mag, mag_sample), "magnetic field accepted");

    perception::ros_msg::FluidPressureMessage pressure_sample;
    pressure_sample.header = {stamp + 2, "dvl_link"};
    pressure_sample.data.fluid_pressure = 301325.75;
    pressure_sample.data.variance = 12.5;
    check(perception::ros_msg::write(recorder, pressure, pressure_sample), "pressure accepted");

    recorder.close();
    check(recorder.written() == 3, "all three sensor messages reached the file");
    check(recorder.drops() == 0, "none dropped");
    check(recorder.grew() == 0, "no sensor encode outgrew its declared max_bytes");
  }

  // What was stored is what should be on the wire: both are row-major.
  auto row_major3 = [&](double scale) {
    const std::array<double, 9> m = asymmetric3(scale);
    return std::vector<double>(m.begin(), m.end());
  };

  mcap::McapReader reader;
  if (!reader.open(path).ok()) {
    check(false, "reopening the sensor file");
    return;
  }
  reader.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);
  check(reader.channels().size() == 3, "three sensor channels");
  check(reader.schemas().size() == 3, "three distinct schemas");

  int decoded = 0;
  for (const auto& message : reader.readMessages()) {
    const std::string& topic = message.channel->topic;
    test::CdrReader cdr(message.message.data, message.message.dataSize);

    const int32_t sec = cdr.i32();
    const uint32_t nsec = cdr.u32();
    const std::string frame_id = cdr.str();
    const uint64_t restamped = static_cast<uint64_t>(sec) * 1000000000ull + nsec;

    if (topic == "/imu/data") {
      ++decoded;
      const std::vector<double> q = cdr.f64_array(4);
      const std::vector<double> qcov = cdr.f64_array(9);
      const std::vector<double> w = cdr.f64_array(3);
      const std::vector<double> wcov = cdr.f64_array(9);
      const std::vector<double> a = cdr.f64_array(3);
      const std::vector<double> acov = cdr.f64_array(9);

      check(frame_id == "imu_link" && restamped == stamp, "imu header");
      check(q == std::vector<double>({0.5, -0.5, 0.5, 0.5}),
            "imu orientation, in ROS x/y/z/w order");
      check(qcov == row_major3(1.0), "orientation covariance, row-major and in order");
      check(w == std::vector<double>({0.125, -0.25, 0.375}), "angular velocity");
      check(wcov == row_major3(2.0), "angular velocity covariance");
      check(a == std::vector<double>({-1.5, 9.75, 0.0625}), "linear acceleration");
      check(acov == row_major3(3.0), "linear acceleration covariance");
      check(cdr.exhausted(), "the Imu message ends exactly where the schema says");

    } else if (topic == "/imu/mag") {
      ++decoded;
      const std::vector<double> field = cdr.f64_array(3);
      const std::vector<double> cov = cdr.f64_array(9);
      check(field == std::vector<double>({1.5e-5, -2.5e-5, 4.75e-5}), "magnetic field vector");
      check(cov == row_major3(1.0), "magnetic field covariance");
      check(cdr.exhausted(), "the MagneticField message ends exactly");

    } else if (topic == "/pressure") {
      ++decoded;
      check(cdr.f64() == 301325.75, "fluid pressure");
      check(cdr.f64() == 12.5, "pressure variance");
      check(cdr.exhausted(), "the FluidPressure message ends exactly");

    }
  }
  check(decoded == 3, "every sensor message was found and decoded");

  schemas_are_self_contained(path);
  reader.close();
  std::filesystem::remove(path);
}

}  // namespace

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "perception_mcap_test";
  std::filesystem::remove_all(root);

  std::string path;
  const uint64_t kStamp = 1788281225865213416ull;  // a real PTP/TAI stamp from the rig

  {
    perception::McapRecorder::Config config;
    config.root = root.string();
    config.compress = false;  // so a failure is a bug here, not in zstd

    perception::McapRecorder recorder(config);
    path = recorder.path();

    std::vector<perception::ros_msg::Topic<perception::ros_msg::ImageMessage>> images;
    for (const char* role : {"left", "right"}) {
      images.push_back(perception::ros_msg::add_topic<perception::ros_msg::ImageMessage>(
          recorder, "/" + std::string(role) + "/image_raw", 3.5));
    }

    perception::ros_msg::DisparityContext disparity_context;
    disparity_context.width = kDispW;
    disparity_context.height = kDispH;
    disparity_context.focal_length_px = 2301.824466f;
    disparity_context.baseline_m = 0.10846109f;
    disparity_context.min_disparity = 0.0f;
    disparity_context.max_disparity = 64.0f;
    const auto disparity = perception::ros_msg::add_topic<perception::ros_msg::DisparityMessage>(
        recorder, "/disparity", 3.5, std::move(disparity_context));

    check(recorder.topic_count() == 3, "three topics declared");
    recorder.start();

    check(perception::ros_msg::write(
              recorder, images[0],
              perception::ros_msg::ImageMessage{
                  {kStamp, "left_optical"},
                  {kImageDesc, make_image(1).data(), kImageW * kImageH}}),
          "left image accepted");
    check(perception::ros_msg::write(
              recorder, images[1],
              perception::ros_msg::ImageMessage{
                  {kStamp + 800, "right_optical"},
                  {kImageDesc, make_image(100).data(), kImageW * kImageH}}),
          "right image accepted");

    // A disparity frame, borrowed the way DownloadStage hands one over.
    const std::size_t disparity_bytes = static_cast<std::size_t>(kDispW) * kDispH * sizeof(float);
    auto storage = std::make_unique<float[]>(static_cast<std::size_t>(kDispW) * kDispH);
    for (std::size_t i = 0; i < static_cast<std::size_t>(kDispW) * kDispH; ++i) {
      storage[i] = 10.0f + static_cast<float>(i) * 0.5f;
    }
    check(perception::ros_msg::write(recorder, disparity,
                                     perception::ros_msg::DisparityMessage{
                                         {kStamp, "left_optical"},
                                         {storage.get(), disparity_bytes}}),
          "disparity accepted");

    recorder.close();
    check(recorder.written() == 3, "all three messages reached the file");
    check(recorder.drops() == 0, "nothing dropped");
    check(recorder.grew() == 0, "no encode outgrew its topic's reserved buffer");
  }

  check(std::filesystem::exists(path) && std::filesystem::file_size(path) > 0,
        "the file exists and is not empty");

  // --- read it back -----------------------------------------------------------
  mcap::McapReader reader;
  const mcap::Status open_status = reader.open(path);
  check(open_status.ok(), "mcap reopens the file it wrote");
  if (!open_status.ok()) {
    std::printf("FAIL\n");
    return 1;
  }

  const mcap::Status summary = reader.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);
  check(summary.ok(), "the summary reads");
  check(reader.channels().size() == 3, "three channels");
  check(reader.schemas().size() == 2, "two schemas (Image, DisparityImage)");

  bool profile_ok = false;
  if (const auto header = reader.header()) profile_ok = header->profile == "ros2";
  check(profile_ok, "the ros2 profile is declared");

  bool image_schema_ok = false;
  bool disparity_schema_ok = false;
  for (const auto& [id, schema] : reader.schemas()) {
    if (schema->name == "sensor_msgs/msg/Image") image_schema_ok = schema->encoding == "ros2msg";
    if (schema->name == "stereo_msgs/msg/DisparityImage") {
      disparity_schema_ok = schema->encoding == "ros2msg";
    }
  }
  check(image_schema_ok, "the Image schema is named and encoded as ROS 2 expects");
  check(disparity_schema_ok, "so is DisparityImage");

  int images_seen = 0;
  bool left_ok = false;
  bool disparity_ok = false;
  bool stamp_ok = false;

  for (const auto& message : reader.readMessages()) {
    const std::string& topic = message.channel->topic;
    test::CdrReader cdr(message.message.data, message.message.dataSize);

    if (topic == "/left/image_raw" || topic == "/right/image_raw") {
      ++images_seen;
      const int32_t sec = cdr.i32();
      const uint32_t nsec = cdr.u32();
      const std::string frame_id = cdr.str();
      const uint32_t height = cdr.u32();
      const uint32_t width = cdr.u32();
      const std::string encoding = cdr.str();
      const uint8_t big_endian = cdr.u8();
      const uint32_t step = cdr.u32();
      const std::vector<unsigned char> data = cdr.bytes();

      if (topic == "/left/image_raw") {
        const uint64_t stamp = static_cast<uint64_t>(sec) * 1000000000ull + nsec;
        stamp_ok = stamp == kStamp;
        left_ok = frame_id == "left_optical" && height == kImageH && width == kImageW &&
                  encoding == "bayer_rggb8" && big_endian == 0 && step == kImageW &&
                  data == make_image(1) && cdr.exhausted();
      }
    } else if (topic == "/disparity") {
      cdr.i32();
      cdr.u32();
      cdr.str();  // outer header
      cdr.i32();
      cdr.u32();
      const std::string inner_frame = cdr.str();  // nested Image header
      const uint32_t height = cdr.u32();
      const uint32_t width = cdr.u32();
      const std::string encoding = cdr.str();
      cdr.u8();
      const uint32_t step = cdr.u32();
      const std::vector<unsigned char> data = cdr.bytes();
      const float f = cdr.f32();
      const float t = cdr.f32();
      const uint32_t x_offset = cdr.u32();
      const uint32_t y_offset = cdr.u32();
      const uint32_t roi_h = cdr.u32();
      const uint32_t roi_w = cdr.u32();
      const bool do_rectify = cdr.b();
      const float min_d = cdr.f32();
      const float max_d = cdr.f32();
      cdr.f32();  // delta_d

      float first = 0.0f;
      float last = 0.0f;
      if (data.size() == static_cast<std::size_t>(kDispW) * kDispH * sizeof(float)) {
        std::memcpy(&first, data.data(), sizeof(float));
        std::memcpy(&last, data.data() + data.size() - sizeof(float), sizeof(float));
      }

      disparity_ok = inner_frame == "left_optical" && height == kDispH && width == kDispW &&
                     encoding == "32FC1" && step == kDispW * 4 && f > 2301.8f && f < 2301.9f &&
                     t > 0.1084f && t < 0.1085f && x_offset == 0 && y_offset == 0 &&
                     roi_h == kDispH && roi_w == kDispW && do_rectify && min_d == 0.0f &&
                     max_d == 64.0f && first == 10.0f &&
                     last == 10.0f + (kDispW * kDispH - 1) * 0.5f && cdr.exhausted();
    }
  }

  check(images_seen == 2, "both images were read back");
  check(stamp_ok, "the PTP stamp survives the sec/nanosec split exactly");
  check(left_ok, "every Image field decodes to what was written");
  check(disparity_ok, "every DisparityImage field decodes, floats included");

  reader.close();

  schemas_are_self_contained(path);
  topics_do_not_evict_each_other(root);
  schema_sharing_and_lifecycle(root);
  the_tail_is_not_lost(root);
  sensor_messages_round_trip(root);

  std::filesystem::remove_all(root);

  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
