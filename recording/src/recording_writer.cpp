#include "recording_writer.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <stdexcept>

namespace perception {
namespace {

uint64_t wall_now_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count());
}

uint64_t steady_now_us() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}

std::string format_time(uint64_t wall_ns, const char* format) {
  const std::time_t seconds = static_cast<std::time_t>(wall_ns / 1'000'000'000ull);
  std::tm tm{};
  gmtime_r(&seconds, &tm);
  std::array<char, 64> buffer{};
  const std::size_t n = std::strftime(buffer.data(), buffer.size(), format, &tm);
  return std::string(buffer.data(), n);
}

// Loops, because a short write is legal on a large buffer and treating one as
// an error would fail a recording that was actually fine.
bool write_all(int fd, const void* data, std::size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  while (bytes > 0) {
    const ssize_t n = ::write(fd, p, bytes);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      return false;
    }
    p += n;
    bytes -= static_cast<std::size_t>(n);
  }
  return true;
}

bool pwrite_all(int fd, const void* data, std::size_t bytes, uint64_t offset) {
  const auto* p = static_cast<const unsigned char*>(data);
  while (bytes > 0) {
    const ssize_t n = ::pwrite(fd, p, bytes, static_cast<off_t>(offset));
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      return false;
    }
    p += n;
    bytes -= static_cast<std::size_t>(n);
    offset += static_cast<uint64_t>(n);
  }
  return true;
}

int open_truncating(const std::string& path) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    throw std::runtime_error("recording: cannot open " + path + ": " + std::strerror(errno));
  }
  return fd;
}

}  // namespace

std::string iso8601_utc(uint64_t wall_ns) { return format_time(wall_ns, "%Y-%m-%dT%H:%M:%SZ"); }

std::string recording_dir_name(uint64_t wall_ns) {
  return "recording-" + format_time(wall_ns, "%Y-%m-%dT%H-%M-%S");
}

// --- StreamWriter ------------------------------------------------------------

StreamWriter::StreamWriter(const std::string& data_path, const std::string& index_path,
                           const Config& config)
    : config_(config), data_path_(data_path) {
  if (config_.frame_bytes == 0) throw std::runtime_error("recording: zero frame_bytes");
  if (config_.staging_frames == 0) throw std::runtime_error("recording: zero staging_frames");

  data_fd_ = open_truncating(data_path);
  try {
    index_fd_ = open_truncating(index_path);
  } catch (...) {
    ::close(data_fd_);
    throw;
  }

  staging_.reserve(config_.staging_frames);
  for (uint32_t i = 0; i < config_.staging_frames; ++i) {
    staging_.emplace_back(new unsigned char[config_.frame_bytes]);
  }
  meta_.resize(config_.staging_frames);

  thread_ = std::thread(&StreamWriter::run, this);
}

StreamWriter::~StreamWriter() { close(); }

std::string StreamWriter::error() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return error_;
}

bool StreamWriter::push(uint64_t timestamp_ns, uint64_t host_recv_ns, uint32_t frame_id,
                        const void* data, std::size_t bytes) {
  if (bytes != config_.frame_bytes) {
    throw std::runtime_error("recording: frame size changed mid-recording");
  }

  std::size_t slot = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_ || !error_.empty() || count_ == staging_.size()) {
      drops_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    slot = (head_ + count_) % staging_.size();
    ++count_;
    if (count_ > staging_peak_.load(std::memory_order_relaxed)) {
      staging_peak_.store(static_cast<uint32_t>(count_), std::memory_order_relaxed);
    }
  }

  // Outside the lock: this is the one expensive thing push() does (~150-300us
  // for a 1.5 MB frame at 60 Hz stereo) and the writer thread has no business
  // waiting on it. Reserving above and publishing below is what keeps the
  // writer off a slot whose copy is still running.
  std::memcpy(staging_[slot].get(), data, bytes);

  meta_[slot].timestamp_ns = timestamp_ns;
  meta_[slot].host_recv_ns = host_recv_ns;
  meta_[slot].bytes = static_cast<uint32_t>(bytes);
  meta_[slot].frame_id = frame_id;
  // offset is assigned by the writer thread, the only thing that knows where
  // the file has got to.

  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++published_;
  }
  queued_.notify_one();
  return true;
}

void StreamWriter::run() {
  for (;;) {
    std::size_t slot = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      queued_.wait(lock, [this] { return !running_ || published_ > 0; });
      // Drains before leaving: a close() that threw away queued frames would
      // lose the tail of every recording.
      if (published_ == 0) break;
      slot = head_;
    }

    const uint64_t started_us = steady_now_us();
    const bool ok = write_frame(meta_[slot], staging_[slot].get());
    const uint64_t elapsed_us = steady_now_us() - started_us;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      head_ = (head_ + 1) % staging_.size();
      --count_;
      --published_;
      if (!ok && error_.empty()) {
        error_ = "write failed on " + data_path_ + ": " + std::strerror(errno);
      }
    }

    if (ok) {
      uint64_t worst = write_max_us_.load(std::memory_order_relaxed);
      while (elapsed_us > worst &&
             !write_max_us_.compare_exchange_weak(worst, elapsed_us, std::memory_order_relaxed)) {
      }
    }
  }
}

bool StreamWriter::write_frame(const IndexRecord& record_in, const void* data) {
  IndexRecord record = record_in;
  record.offset = offset_;

  // pwrite at the padded offset rather than write-then-pad: the padding is
  // never written at all, it is a hole, and the file is ftruncated to the right
  // length at close. Costs nothing and saves 0.08% of the write bandwidth.
  if (!pwrite_all(data_fd_, data, record.bytes, record.offset)) return false;
  if (!write_all(index_fd_, &record, sizeof(record))) return false;

  offset_ += round_up(record.bytes, kRecordAlign);

  if (written_.load(std::memory_order_relaxed) == 0) {
    first_ts_.store(record.timestamp_ns, std::memory_order_relaxed);
  }
  last_ts_.store(record.timestamp_ns, std::memory_order_relaxed);
  bytes_written_.fetch_add(record.bytes, std::memory_order_relaxed);
  written_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void StreamWriter::close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) return;
    running_ = false;
  }
  queued_.notify_all();
  if (thread_.joinable()) thread_.join();

  if (data_fd_ >= 0) {
    // The last frame's padding is a hole that was never written, so without
    // this the file is short of its own record_stride_bytes grid.
    if (::ftruncate(data_fd_, static_cast<off_t>(offset_)) != 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (error_.empty()) error_ = "ftruncate failed on " + data_path_;
    }
    ::close(data_fd_);
    data_fd_ = -1;
  }
  if (index_fd_ >= 0) {
    ::close(index_fd_);
    index_fd_ = -1;
  }
}

// --- RecordingWriter ---------------------------------------------------------

RecordingWriter::RecordingWriter(const Config& config, const std::vector<StreamInfo>& streams)
    : config_(config) {
  if (streams.empty()) throw std::runtime_error("recording: no streams");

  started_ns_ = wall_now_ns();
  const std::filesystem::path root(config_.root);
  const std::filesystem::path dir = root / recording_dir_name(started_ns_);

  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    throw std::runtime_error("recording: cannot create " + dir.string() + ": " + ec.message());
  }
  directory_ = dir.string();

  manifest_.version = kRecordingVersion;
  manifest_.created_utc = iso8601_utc(started_ns_);

  for (std::size_t i = 0; i < streams.size(); ++i) {
    StreamInfo info = streams[i];
    info.id = static_cast<uint32_t>(i);
    info.record_stride_bytes = round_up(info.frame_bytes, kRecordAlign);
    info.data = "cam" + std::to_string(i) + ".dat";
    info.index = "cam" + std::to_string(i) + ".idx";
    info.frames = 0;

    StreamWriter::Config writer_config;
    writer_config.frame_bytes = info.frame_bytes;
    writer_config.staging_frames = config_.staging_frames;

    writers_.emplace_back(new StreamWriter((dir / info.data).string(),
                                           (dir / info.index).string(), writer_config));
    manifest_.streams.push_back(std::move(info));
  }
}

RecordingWriter::~RecordingWriter() { close(); }

bool RecordingWriter::push(uint32_t stream, uint64_t timestamp_ns, uint64_t host_recv_ns,
                           uint32_t frame_id, const void* data, std::size_t bytes) {
  if (stream >= writers_.size()) throw std::runtime_error("recording: bad stream id");
  return writers_[stream]->push(timestamp_ns, host_recv_ns, frame_id, data, bytes);
}

void RecordingWriter::close() {
  if (closed_) return;
  closed_ = true;

  for (auto& writer : writers_) writer->close();

  // The shared epoch is the minimum first timestamp across streams, not each
  // stream's own first frame: a camera that started a second late has to replay
  // a second late, and a per-stream origin would silently align them.
  uint64_t epoch = 0;
  for (std::size_t i = 0; i < writers_.size(); ++i) {
    manifest_.streams[i].frames = writers_[i]->written();
    const uint64_t first = writers_[i]->first_timestamp_ns();
    if (first != 0 && (epoch == 0 || first < epoch)) epoch = first;
  }
  manifest_.epoch_ns = epoch;

  write_manifest((std::filesystem::path(directory_) / "manifest.yaml").string(), manifest_);
}

std::string RecordingWriter::health_line() const {
  std::string drops;
  uint64_t total_bytes = 0;
  uint32_t peak = 0;
  uint64_t worst_us = 0;
  for (const auto& writer : writers_) {
    if (!drops.empty()) drops += "/";
    drops += std::to_string(writer->drops());
    total_bytes += writer->bytes_written();
    peak = std::max(peak, writer->staging_peak());
    worst_us = std::max(worst_us, writer->write_max_us());
  }

  const double elapsed_s = static_cast<double>(wall_now_ns() - started_ns_) * 1e-9;
  const double mb_s = elapsed_s > 0.0 ? static_cast<double>(total_bytes) / elapsed_s / 1e6 : 0.0;

  char buffer[192];
  std::snprintf(buffer, sizeof(buffer), "rec drops=%s staging_peak=%u write_max=%luus %.1fMB/s",
                drops.c_str(), peak, static_cast<unsigned long>(worst_us), mb_s);
  return buffer;
}

void write_manifest(const std::string& path, const RecordingManifest& manifest) {
  YAML::Emitter out;
  out << YAML::BeginMap;

  out << YAML::Key << "recording" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "version" << YAML::Value << manifest.version;
  out << YAML::Key << "created_utc" << YAML::Value << manifest.created_utc;
  out << YAML::Key << "epoch_ns" << YAML::Value << manifest.epoch_ns;
  out << YAML::Key << "ptp_status_at_start" << YAML::Value << manifest.ptp_status_at_start;
  out << YAML::EndMap;

  out << YAML::Key << "streams" << YAML::Value << YAML::BeginSeq;
  for (const StreamInfo& stream : manifest.streams) {
    out << YAML::BeginMap;
    out << YAML::Key << "id" << YAML::Value << stream.id;
    out << YAML::Key << "role" << YAML::Value << stream.role;
    out << YAML::Key << "serial" << YAML::Value << stream.serial;
    out << YAML::Key << "width" << YAML::Value << stream.width;
    out << YAML::Key << "height" << YAML::Value << stream.height;
    out << YAML::Key << "stride_bytes" << YAML::Value << stream.stride_bytes;
    out << YAML::Key << "frame_bytes" << YAML::Value << stream.frame_bytes;
    out << YAML::Key << "record_stride_bytes" << YAML::Value << stream.record_stride_bytes;
    out << YAML::Key << "pixel_format" << YAML::Value << stream.pixel_format;
    out << YAML::Key << "frames" << YAML::Value << stream.frames;
    out << YAML::Key << "data" << YAML::Value << stream.data;
    out << YAML::Key << "index" << YAML::Value << stream.index;
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;

  // Provenance. A sequence of single-key maps, matching the camera.features
  // spelling in the config this came from, so it can be pasted straight back.
  out << YAML::Key << "camera_config" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "features" << YAML::Value << YAML::BeginSeq;
  for (const auto& [name, value] : manifest.camera_features) {
    out << YAML::BeginMap << YAML::Key << name << YAML::Value << value << YAML::EndMap;
  }
  out << YAML::EndSeq << YAML::EndMap;

  out << YAML::EndMap;

  const int fd = open_truncating(path);
  const bool ok = write_all(fd, out.c_str(), static_cast<std::size_t>(out.size())) &&
                  write_all(fd, "\n", 1);
  ::close(fd);
  if (!ok) throw std::runtime_error("recording: cannot write " + path);
}

}  // namespace perception
