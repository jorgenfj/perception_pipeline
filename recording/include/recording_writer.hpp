#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "recording_format.hpp"

namespace perception {

// One stream's writer: a bounded staging ring in front of a writer thread.
//
// push() copies the frame and returns. It never blocks on the disk and never
// holds the caller's buffer past the memcpy, which is the whole point: the
// recorder must not be able to turn a slow disk into camera backpressure, or
// enabling recording changes the timing of the run being recorded and
// manufactures the exact frame_id gaps the index format exists to identify.
// See recording_plan.md, "Backpressure".
//
// The price is that a disk that cannot keep up drops frames instead of slowing
// the camera. That is deliberate and it is visible: drops() is the recorder's
// health number and it is local to the recorder.
class StreamWriter {
 public:
  struct Config {
    // Payload size per frame. Every push() must match this.
    std::size_t frame_bytes = 0;

    // Depth of the staging ring. 32 frames is ~50 MB at 1440x1080 Bayer8 --
    // enough to ride out a page-cache flush, bounded so a stalled disk costs
    // memory that was budgeted rather than all of it.
    uint32_t staging_frames = 32;
  };

  // Creates (truncating) both files. Throws if either cannot be opened.
  StreamWriter(const std::string& data_path, const std::string& index_path, const Config& config);
  ~StreamWriter();

  StreamWriter(const StreamWriter&) = delete;
  StreamWriter& operator=(const StreamWriter&) = delete;

  // Copy the frame into the staging ring. False means the ring was full and the
  // frame was dropped -- the caller does nothing about it, that is what the
  // counter is for. `bytes` must equal config.frame_bytes.
  bool push(uint64_t timestamp_ns, uint64_t host_recv_ns, uint32_t frame_id, const void* data,
            std::size_t bytes);

  // Drain the staging ring, join the writer thread and close the files.
  // Idempotent; also called by the destructor.
  void close();

  uint64_t written() const { return written_.load(std::memory_order_relaxed); }
  uint64_t drops() const { return drops_.load(std::memory_order_relaxed); }
  uint64_t bytes_written() const { return bytes_written_.load(std::memory_order_relaxed); }

  // High-water mark of the staging ring. Approaching staging_frames is the
  // warning that arrives before drops start.
  uint32_t staging_peak() const { return staging_peak_.load(std::memory_order_relaxed); }

  // Worst single frame write, microseconds -- the disk's tail latency.
  uint64_t write_max_us() const { return write_max_us_.load(std::memory_order_relaxed); }

  // First and last camera timestamp actually written. The manifest's shared
  // epoch is the minimum of first_timestamp_ns() across streams.
  uint64_t first_timestamp_ns() const { return first_ts_.load(std::memory_order_relaxed); }
  uint64_t last_timestamp_ns() const { return last_ts_.load(std::memory_order_relaxed); }

  // Non-empty once a write has failed. The writer thread stops writing at that
  // point and everything after is counted as a drop, so a full disk ends up as
  // a truncated recording plus a message rather than a crash mid-run.
  std::string error() const;

 private:
  void run();
  bool write_frame(const IndexRecord& record, const void* data);

  Config config_;
  std::string data_path_;
  int data_fd_ = -1;
  int index_fd_ = -1;

  // Staging slots, each one frame payload. Allocated up front: a recorder that
  // allocates per frame is a recorder that stalls on the allocator.
  std::vector<std::unique_ptr<unsigned char[]>> staging_;
  std::vector<IndexRecord> meta_;

  // Single producer (the stream's acquisition thread), single consumer (the
  // writer thread), so the two counters below are enough: reservation order is
  // publication order, and the published slots are always the `published_` of
  // them starting at head_.
  mutable std::mutex mutex_;
  std::condition_variable queued_;
  std::size_t head_ = 0;       // next slot the writer will take
  std::size_t count_ = 0;      // slots reserved, copy possibly still running
  std::size_t published_ = 0;  // of those, how many are fully copied
  bool running_ = true;
  std::string error_;

  // Where the next payload goes in the .dat. Only the writer thread touches it.
  uint64_t offset_ = 0;

  std::thread thread_;
  std::atomic<uint64_t> written_{0};
  std::atomic<uint64_t> drops_{0};
  std::atomic<uint64_t> bytes_written_{0};
  std::atomic<uint32_t> staging_peak_{0};
  std::atomic<uint64_t> write_max_us_{0};
  std::atomic<uint64_t> first_ts_{0};
  std::atomic<uint64_t> last_ts_{0};
};

// The whole recording: a directory, one StreamWriter per camera, and the
// manifest that describes them. Streams are otherwise independent -- nothing
// here correlates them, because the file has no opinion about pairing.
class RecordingWriter {
 public:
  struct Config {
    // Parent directory. A timestamped subdirectory is created inside it.
    std::string root = "recordings";
    uint32_t staging_frames = 32;
  };

  // `streams` supplies role, serial and geometry; the id, file names and
  // record_stride_bytes are filled in here so there is one place that decides
  // them. Creates the directory and opens every stream's files.
  RecordingWriter(const Config& config, const std::vector<StreamInfo>& streams);
  ~RecordingWriter();

  RecordingWriter(const RecordingWriter&) = delete;
  RecordingWriter& operator=(const RecordingWriter&) = delete;

  // Provenance for the manifest. Set before close(); ignored after.
  void set_camera_features(std::vector<std::pair<std::string, std::string>> features) {
    manifest_.camera_features = std::move(features);
  }
  void set_ptp_status(std::string status) { manifest_.ptp_status_at_start = std::move(status); }

  bool push(uint32_t stream, uint64_t timestamp_ns, uint64_t host_recv_ns, uint32_t frame_id,
            const void* data, std::size_t bytes);

  // Drains every stream, then writes manifest.yaml. Written last on purpose:
  // it carries the per-stream frame counts and the shared epoch, neither of
  // which is known until the writers have stopped. A recording whose manifest
  // is missing is one that was killed mid-run, and that is a useful thing for
  // it to look like. Idempotent.
  void close();

  const std::string& directory() const { return directory_; }
  const StreamWriter& stream(uint32_t id) const { return *writers_.at(id); }
  std::size_t stream_count() const { return writers_.size(); }

  // "rec drops=0/0 staging_peak=3 write_max=812us 46.7MB/s"
  std::string health_line() const;

 private:
  Config config_;
  std::string directory_;
  RecordingManifest manifest_;
  std::vector<std::unique_ptr<StreamWriter>> writers_;
  uint64_t started_ns_ = 0;
  bool closed_ = false;
};

// Serialise a manifest to `path`. Exposed for tests, which want to round-trip
// it without a camera anywhere in the picture.
void write_manifest(const std::string& path, const RecordingManifest& manifest);

}  // namespace perception
