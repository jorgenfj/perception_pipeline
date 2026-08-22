#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "frame_source.hpp"
#include "recording_reader.hpp"

namespace perception {

// One stream of a recording, replayed into a FrameSink at the pacing it was
// captured with, so the GPU pipeline can be run and profiled at a desk on
// frames that came off the real cameras.
//
// RecordingReader::read_frame() takes a destination pointer, so a frame goes
// from the .dat into the pinned ingress buffer with no staging copy in between.
//
// One stream, not two: replaying both halves of a stereo recording into the GPU
// stereo consumer is a different job and does not belong behind this interface.
class RecordingSource final : public FrameSource {
 public:
  struct Config {
    // Recording directory -- the one holding manifest.yaml.
    std::string directory;

    // `role` wins when set ("left" / "right", matched against the manifest);
    // otherwise the index is used. The role is the safer form -- stream order
    // is just the order the recorder opened the cameras in.
    uint32_t stream = 0;
    std::string role;

    // Playback rate. Above 1.0 this is a load knob as much as a time one: it
    // raises the frame rate the pipeline has to keep up with.
    double speed = 1.0;

    // Restart at the end rather than finishing the run.
    bool loop = true;

    // Emit timestamps shifted onto the current wall clock, preserving every
    // interval exactly. LatencyProbe measures latency as
    // host_now_ns() - timestamp_ns, so without this a recording made last week
    // reports a week of latency and every latency number becomes noise. Off
    // pushes the file's original stamps through verbatim.
    bool rebase_timestamps = true;

    // How long a frame waits for a free slot before it is given up on. Drops
    // here mean the pipeline is not keeping up with the recorded rate, which is
    // a finding rather than something to paper over by blocking.
    uint64_t slot_wait_ms = 500;
  };

  // Opens the recording and resolves the stream, so geometry() is readable
  // immediately. Throws if the directory, manifest or named stream is missing.
  explicit RecordingSource(const Config& config);
  ~RecordingSource() override;

  RecordingSource(const RecordingSource&) = delete;
  RecordingSource& operator=(const RecordingSource&) = delete;

  const CameraGeometry& geometry() const override { return geometry_; }

  // One in flight, one to fill.
  uint32_t min_slot_count() const override { return 2; }

  void start(FrameSink& sink) override;
  void stop() override;

  bool finished() const override { return finished_.load(std::memory_order_acquire); }
  bool failed() const override { return failed_.load(std::memory_order_acquire); }
  const std::string& failure() const override { return failure_; }

  void set_finished_callback(std::function<void()> cb) override {
    on_finished_ = std::move(cb);
  }

  uint64_t delivered() const override { return delivered_.load(std::memory_order_relaxed); }

  // "late=12 slot_drops=0 loops=3"
  std::string counters() const override;
  std::string notes() const override;

  // What the manifest says PTP was doing when this was recorded. Provenance,
  // and labelled as such -- there is no clock being disciplined here.
  std::string ptp_status() override;

  // Frames that had to wait for a slot to come free. Non-zero means the
  // pipeline is running slower than the recording was captured at.
  uint64_t late() const { return late_.load(std::memory_order_relaxed); }

  // Frames given up on because no ingress slot came free within slot_wait_ms.
  // Not just "dropped": the report line already carries UploadStage::dropped().
  uint64_t slot_drops() const { return slot_drops_.load(std::memory_order_relaxed); }

  uint64_t loops() const { return loops_.load(std::memory_order_relaxed); }

  // The recording behind this source, for anything that wants the manifest.
  const RecordingReader& reader() const { return *reader_; }
  uint32_t stream() const { return stream_; }

 private:
  void run(FrameSink& sink);
  void finish(std::string failure_reason);

  // Hand back every slot the reader has finished with, then take a free one;
  // kNoSlot if none came free in time. The sink's consumed() is level-triggered,
  // so polling it is the contract rather than a workaround.
  void reclaim(FrameSink& sink);
  uint32_t acquire_slot(FrameSink& sink, bool& waited);

  Config config_;
  std::unique_ptr<RecordingReader> reader_;
  uint32_t stream_ = 0;
  CameraGeometry geometry_;

  // Indexed by sink slot: true from commit() until consumed() says the read
  // retired. Only the playback thread touches it.
  std::vector<bool> held_;

  std::thread thread_;
  std::atomic<bool> running_{false};

  std::string failure_;
  std::atomic<bool> failed_{false};
  std::atomic<bool> finished_{false};
  std::function<void()> on_finished_;

  std::atomic<uint64_t> delivered_{0};
  std::atomic<uint64_t> late_{0};
  std::atomic<uint64_t> slot_drops_{0};
  std::atomic<uint64_t> loops_{0};
};

}  // namespace perception
