// Round-trip tests for the recording format: write frames, read them back, and
// check that what comes out is what went in -- pixels, timestamps and pairing.
//
// No GPU and no camera. The one thing it does need is a filesystem, so it works
// in a temporary directory and removes it afterwards.
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "recording_reader.hpp"
#include "recording_writer.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++failures;
}

using perception::IndexRecord;
using perception::RecordingReader;
using perception::RecordingWriter;
using perception::StreamInfo;

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 32;
constexpr std::size_t kFrameBytes = kWidth * kHeight;
constexpr uint64_t kPeriod = 16'666'667;
constexpr uint64_t kEpoch = 1'700'000'000'000'000'000ull;

std::vector<StreamInfo> stereo_streams() {
  std::vector<StreamInfo> streams;
  for (int i = 0; i < 2; ++i) {
    StreamInfo info;
    info.role = i == 0 ? "left" : "right";
    info.serial = i == 0 ? "1111" : "2222";
    info.width = kWidth;
    info.height = kHeight;
    info.stride_bytes = kWidth;
    info.frame_bytes = kFrameBytes;
    info.pixel_format = "BayerRG8";
    streams.push_back(std::move(info));
  }
  return streams;
}

// A frame whose every byte is a function of (stream, index), so a payload that
// came back from the wrong offset is obvious rather than plausible.
std::vector<unsigned char> make_frame(uint32_t stream, uint32_t index) {
  std::vector<unsigned char> frame(kFrameBytes);
  for (std::size_t i = 0; i < frame.size(); ++i) {
    frame[i] = static_cast<unsigned char>((i * 31 + index * 7 + stream * 101) & 0xff);
  }
  return frame;
}

std::string write_recording(const std::string& root, uint32_t frames, int64_t skew_ns,
                            bool drop_one) {
  RecordingWriter::Config config;
  config.root = root;
  config.staging_frames = 8;

  RecordingWriter writer(config, stereo_streams());
  writer.set_ptp_status("Slave");
  writer.set_camera_features({{"PixelFormat", "BayerRG8"}, {"ExposureTime", "8000.0"}});

  for (uint32_t i = 0; i < frames; ++i) {
    const std::vector<unsigned char> left = make_frame(0, i);
    const std::vector<unsigned char> right = make_frame(1, i);
    const uint64_t timestamp = kEpoch + i * kPeriod;

    writer.push(0, timestamp, timestamp + 2'000'000, i, left.data(), left.size());
    // One frame missing from the right stream: a dropout is a gap and nothing
    // in the format describes it, which is exactly what this checks.
    if (!(drop_one && i == 2)) {
      writer.push(1, static_cast<uint64_t>(static_cast<int64_t>(timestamp) + skew_ns),
                  timestamp + 2'000'000, i, right.data(), right.size());
    }
  }

  writer.close();
  return writer.directory();
}

void test_round_trip() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "recording_test_round_trip";
  std::filesystem::remove_all(root);

  const std::string dir = write_recording(root.string(), 6, 120'000, false);
  RecordingReader reader(dir);

  check(reader.stream_count() == 2, "both streams are in the manifest");
  check(reader.manifest().epoch_ns == kEpoch,
        "the epoch is the earliest first timestamp across streams");
  check(reader.manifest().ptp_status_at_start == "Slave", "provenance survives the round trip");
  check(reader.manifest().camera_features.size() == 2, "so do the camera features");
  check(reader.stream(0).role == "left" && reader.stream(1).role == "right",
        "roles are preserved in order");
  check(reader.stream(0).record_stride_bytes == perception::kRecordAlign,
        "payloads are laid out on the 4096 grid");

  check(reader.index(0).size() == 6 && reader.index(1).size() == 6, "every frame is indexed");

  bool payloads_ok = true;
  bool stamps_ok = true;
  std::vector<unsigned char> got(kFrameBytes);
  for (uint32_t s = 0; s < 2; ++s) {
    for (uint32_t i = 0; i < 6; ++i) {
      reader.read_frame(s, i, got.data());
      const std::vector<unsigned char> want = make_frame(s, i);
      if (std::memcmp(got.data(), want.data(), kFrameBytes) != 0) payloads_ok = false;

      const IndexRecord& record = reader.index(s)[i];
      if (record.frame_id != i) stamps_ok = false;
      if (record.bytes != kFrameBytes) stamps_ok = false;
      if (record.offset != static_cast<uint64_t>(i) * perception::kRecordAlign) stamps_ok = false;
      if (record.host_recv_ns <= record.timestamp_ns) stamps_ok = false;
    }
  }
  check(payloads_ok, "every payload reads back byte for byte");
  check(stamps_ok, "the index carries both stamps, the frame id and the right offset");

  std::filesystem::remove_all(root);
}

void test_pairing_is_read_time() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "recording_test_pairing";
  std::filesystem::remove_all(root);

  const std::string dir = write_recording(root.string(), 6, 120'000, true);
  RecordingReader reader(dir);

  // Nothing in the file says what pairs with what, so the answer depends only
  // on the tolerance the reader chooses -- which is the whole argument for
  // keeping pair_id out of the format.
  const perception::PairResult tight = reader.pair(500'000);
  check(tight.stats.paired == 5, "five pairs at 500us, with the dropped frame unpaired");
  check(tight.stats.unpaired[0] == 1 && tight.stats.unpaired[1] == 0,
        "the frame whose partner was never written is the unpaired one");
  check(tight.stats.max_abs_skew_ns == 120'000, "the recorded skew comes back");

  const perception::PairResult loose = reader.pair(50'000);
  check(loose.stats.paired == 0,
        "the same file re-paired at 50us yields nothing: a read-time parameter");

  std::filesystem::remove_all(root);
}

void test_truncated_recording_is_readable() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "recording_test_truncated";
  std::filesystem::remove_all(root);

  const std::string dir = write_recording(root.string(), 6, 0, false);

  // Lop the last two index records off, as a killed run would leave them.
  const std::filesystem::path index = std::filesystem::path(dir) / "cam0.idx";
  std::filesystem::resize_file(index, 4 * sizeof(IndexRecord));

  RecordingReader reader(dir);
  check(reader.index(0).size() == 4,
        "the index length wins over the manifest's count, so a killed run still opens");

  std::vector<unsigned char> got(kFrameBytes);
  reader.read_frame(0, 3, got.data());
  check(std::memcmp(got.data(), make_frame(0, 3).data(), kFrameBytes) == 0,
        "the frames that were written are still readable");

  std::filesystem::remove_all(root);
}

}  // namespace

int main() {
  test_round_trip();
  test_pairing_is_read_time();
  test_truncated_recording_is_readable();

  std::printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
