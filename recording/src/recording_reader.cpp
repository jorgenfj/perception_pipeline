#include "recording_reader.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace perception {
namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("recording: " + what);
}

template <typename T>
T require(const YAML::Node& parent, const char* key, const std::string& where) {
  if (!parent[key]) fail(where + ": missing '" + std::string(key) + "'");
  try {
    return parent[key].as<T>();
  } catch (const YAML::Exception&) {
    fail(where + "." + key + ": cannot read '" + YAML::Dump(parent[key]) + "'");
  }
}

std::vector<unsigned char> read_whole_file(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) fail("cannot open " + path + ": " + std::strerror(errno));

  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    fail("cannot stat " + path);
  }

  std::vector<unsigned char> out(static_cast<std::size_t>(st.st_size));
  std::size_t done = 0;
  while (done < out.size()) {
    const ssize_t n = ::read(fd, out.data() + done, out.size() - done);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      ::close(fd);
      fail("short read on " + path);
    }
    done += static_cast<std::size_t>(n);
  }
  ::close(fd);
  return out;
}

}  // namespace

RecordingManifest read_manifest(const std::string& path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::Exception& e) {
    fail("cannot load " + path + ": " + e.what());
  }

  RecordingManifest manifest;

  const YAML::Node recording = root["recording"];
  if (!recording) fail(path + ": no 'recording' section");
  manifest.version = require<uint32_t>(recording, "version", "recording");
  if (manifest.version != kRecordingVersion) {
    fail(path + ": version " + std::to_string(manifest.version) + ", this build reads version " +
         std::to_string(kRecordingVersion));
  }
  manifest.created_utc = require<std::string>(recording, "created_utc", "recording");
  manifest.epoch_ns = require<uint64_t>(recording, "epoch_ns", "recording");
  if (recording["ptp_status_at_start"]) {
    manifest.ptp_status_at_start = recording["ptp_status_at_start"].as<std::string>();
  }

  const YAML::Node streams = root["streams"];
  if (!streams || !streams.IsSequence() || streams.size() == 0) {
    fail(path + ": 'streams' must be a non-empty sequence");
  }
  for (std::size_t i = 0; i < streams.size(); ++i) {
    const YAML::Node node = streams[i];
    const std::string where = "streams[" + std::to_string(i) + "]";
    StreamInfo info;
    info.id = require<uint32_t>(node, "id", where);
    info.role = require<std::string>(node, "role", where);
    if (node["serial"]) info.serial = node["serial"].as<std::string>();
    info.width = require<uint32_t>(node, "width", where);
    info.height = require<uint32_t>(node, "height", where);
    info.stride_bytes = require<uint32_t>(node, "stride_bytes", where);
    info.frame_bytes = require<std::size_t>(node, "frame_bytes", where);
    info.record_stride_bytes = require<std::size_t>(node, "record_stride_bytes", where);
    info.pixel_format = require<std::string>(node, "pixel_format", where);
    info.frames = require<uint64_t>(node, "frames", where);
    info.data = require<std::string>(node, "data", where);
    info.index = require<std::string>(node, "index", where);
    manifest.streams.push_back(std::move(info));
  }

  // Provenance, read back so `stereo_view --info` can print what the rig was
  // set to. Nothing branches on it.
  const YAML::Node features = root["camera_config"] ? root["camera_config"]["features"]
                                                    : YAML::Node();
  if (features && features.IsSequence()) {
    for (const YAML::Node& entry : features) {
      if (!entry.IsMap() || entry.size() != 1) continue;
      const auto it = entry.begin();
      manifest.camera_features.emplace_back(it->first.Scalar(), it->second.Scalar());
    }
  }

  return manifest;
}

RecordingReader::RecordingReader(const std::string& directory) : directory_(directory) {
  const std::filesystem::path dir(directory);
  manifest_ = read_manifest((dir / "manifest.yaml").string());

  indices_.resize(manifest_.streams.size());
  timestamps_.resize(manifest_.streams.size());
  data_fds_.assign(manifest_.streams.size(), -1);

  for (std::size_t s = 0; s < manifest_.streams.size(); ++s) {
    const StreamInfo& info = manifest_.streams[s];

    const std::vector<unsigned char> raw = read_whole_file((dir / info.index).string());
    if (raw.size() % sizeof(IndexRecord) != 0) {
      fail(info.index + ": size is not a whole number of 32-byte records");
    }

    const std::size_t count = raw.size() / sizeof(IndexRecord);
    indices_[s].resize(count);
    std::memcpy(indices_[s].data(), raw.data(), raw.size());

    // The manifest's count and the index's length can disagree only if the
    // run was killed between the last write and the manifest, so the index
    // wins: it is the thing that was appended to as frames landed.
    timestamps_[s].reserve(count);
    for (const IndexRecord& record : indices_[s]) timestamps_[s].push_back(record.timestamp_ns);

    data_fds_[s] = ::open((dir / info.data).string().c_str(), O_RDONLY);
    if (data_fds_[s] < 0) {
      fail("cannot open " + info.data + ": " + std::strerror(errno));
    }
  }
}

RecordingReader::~RecordingReader() {
  for (int fd : data_fds_) {
    if (fd >= 0) ::close(fd);
  }
}

void RecordingReader::read_frame(std::size_t s, std::size_t i, void* dst) const {
  const std::vector<IndexRecord>& index = indices_.at(s);
  if (i >= index.size()) fail("frame index out of range");

  const IndexRecord& record = index[i];
  auto* out = static_cast<unsigned char*>(dst);
  std::size_t done = 0;
  while (done < record.bytes) {
    const ssize_t n = ::pread(data_fds_.at(s), out + done, record.bytes - done,
                              static_cast<off_t>(record.offset + done));
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      fail("short read on " + manifest_.streams[s].data + " frame " + std::to_string(i) +
           " -- the recording is truncated");
    }
    done += static_cast<std::size_t>(n);
  }
}

PairResult RecordingReader::pair(uint64_t tolerance_ns) const {
  if (manifest_.streams.size() != 2) {
    fail("pairing needs exactly two streams, this recording has " +
         std::to_string(manifest_.streams.size()));
  }

  for (std::size_t s = 0; s < 2; ++s) {
    const std::size_t at = first_timestamp_regression(timestamps_[s].data(), timestamps_[s].size());
    if (at < timestamps_[s].size()) {
      fail("stream " + std::to_string(s) + " timestamp goes backwards at frame " +
           std::to_string(at) + " -- the camera clock was free-running and reset, so the two "
           "cameras share no epoch and pairing them would be meaningless");
    }
  }

  return pair_by_timestamp(timestamps_[0], timestamps_[1], tolerance_ns);
}

bool RecordingReader::is_reconnect_seam(std::size_t s, std::size_t i) const {
  const std::vector<IndexRecord>& index = indices_.at(s);
  if (i == 0 || i >= index.size()) return false;
  return index[i].frame_id < index[i - 1].frame_id &&
         index[i].host_recv_ns >= index[i - 1].host_recv_ns;
}

std::string RecordingReader::info(uint64_t tolerance_ns) const {
  std::string out;
  char line[512];

  std::snprintf(line, sizeof(line), "recording %s  created %s  epoch %luns\n", directory_.c_str(),
                manifest_.created_utc.c_str(), static_cast<unsigned long>(manifest_.epoch_ns));
  out += line;
  if (!manifest_.ptp_status_at_start.empty()) {
    std::snprintf(line, sizeof(line), "ptp at start: %s (provenance only)\n",
                  manifest_.ptp_status_at_start.c_str());
    out += line;
  }

  for (std::size_t s = 0; s < manifest_.streams.size(); ++s) {
    const StreamInfo& info = manifest_.streams[s];
    const std::vector<IndexRecord>& index = indices_[s];

    double duration_s = 0.0;
    if (index.size() >= 2) {
      duration_s = static_cast<double>(index.back().timestamp_ns - index.front().timestamp_ns) *
                   1e-9;
    }

    // A gap is two adjacent records further apart than one and a half frame
    // periods -- nothing in the file describes a dropout, because nothing needs
    // to: the timestamps already do. Half a period of slack, rather than a
    // whole one: a single missing frame leaves a delta of exactly two periods,
    // and a threshold at 2x would be the one thing that misses it.
    std::vector<uint64_t> deltas;
    deltas.reserve(index.size());
    for (std::size_t i = 1; i < index.size(); ++i) {
      deltas.push_back(index[i].timestamp_ns - index[i - 1].timestamp_ns);
    }
    uint64_t median = 0;
    if (!deltas.empty()) {
      std::vector<uint64_t> sorted = deltas;
      std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
      median = sorted[sorted.size() / 2];
    }
    std::size_t gaps = 0;
    std::size_t seams = 0;
    for (std::size_t i = 1; i < index.size(); ++i) {
      if (median != 0 && deltas[i - 1] > median + median / 2) ++gaps;
      if (is_reconnect_seam(s, i)) ++seams;
    }

    // Transport latency, and whether it drifts. Two free-running crystals walk;
    // a locked rig gives a flat, noisy band. A slope is a positive diagnosis of
    // PTP not disciplining -- and it is a measurement, not a stored claim.
    double first_lat_ms = 0.0;
    double last_lat_ms = 0.0;
    if (!index.empty()) {
      first_lat_ms = static_cast<double>(static_cast<int64_t>(index.front().host_recv_ns) -
                                         static_cast<int64_t>(index.front().timestamp_ns)) *
                     1e-6;
      last_lat_ms = static_cast<double>(static_cast<int64_t>(index.back().host_recv_ns) -
                                        static_cast<int64_t>(index.back().timestamp_ns)) *
                    1e-6;
    }

    std::snprintf(line, sizeof(line),
                  "  stream %zu (%s) %s %ux%u %s: %zu frames, %.2fs, %.2f Hz, gaps=%zu "
                  "reconnects=%zu\n",
                  s, info.role.c_str(), info.serial.empty() ? "-" : info.serial.c_str(), info.width,
                  info.height, info.pixel_format.c_str(), index.size(), duration_s,
                  duration_s > 0.0 ? static_cast<double>(index.size()) / duration_s : 0.0, gaps,
                  seams);
    out += line;
    std::snprintf(line, sizeof(line), "    host_recv - timestamp: %.3fms -> %.3fms (drift %.3fms)\n",
                  first_lat_ms, last_lat_ms, last_lat_ms - first_lat_ms);
    out += line;
  }

  if (manifest_.streams.size() == 2) {
    const PairResult result = pair(tolerance_ns);
    std::snprintf(line, sizeof(line),
                  "  pairing at %luus: paired=%lu unpaired=%lu/%lu max_skew=%.1fus\n",
                  static_cast<unsigned long>(tolerance_ns / 1000),
                  static_cast<unsigned long>(result.stats.paired),
                  static_cast<unsigned long>(result.stats.unpaired[0]),
                  static_cast<unsigned long>(result.stats.unpaired[1]),
                  static_cast<double>(result.stats.max_abs_skew_ns) * 1e-3);
    out += line;
  }

  return out;
}

}  // namespace perception
