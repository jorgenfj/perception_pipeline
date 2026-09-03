#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "frame_source.hpp"

namespace perception {

/**
 * @brief One image topic of an MCAP, replayed into a FrameSink at the pacing it
 * was recorded with, so the GPU pipeline can be run and profiled at a desk on
 * frames that came off the real cameras.
 *
 * The counterpart of McapRecorder, and the reason `acquire` still builds and
 * runs on a machine with no rig attached (-DPERCEPTION_SOURCE=recording).
 *
 * One topic, not two. Replaying both halves of a stereo recording into the GPU
 * stereo consumer is a different job and does not belong behind this interface.
 *
 * Pixels are decoded straight out of the message the reader holds and copied
 * once, into the sink's slot -- so a frame costs the same memcpy it would from
 * a camera's DMA target, not a staging buffer on top.
 */
class McapReplaySource final : public FrameSource {
 public:
  struct Config {
    /** The .mcap file to replay. */
    std::string path;

    /**
     * Which image topic to feed the pipeline. `topic` wins when set; otherwise
     * "/<role>/image_raw" is used, and an empty role takes the first image
     * topic in the file. A file with two eyes in it replays one of them.
     */
    std::string role;
    std::string topic;

    /**
     * Playback rate. Above 1.0 this is a load knob as much as a time one: it
     * raises the frame rate the pipeline has to keep up with.
     */
    double speed = 1.0;

    /** Restart at the end rather than finishing the run. */
    bool loop = true;

    /**
     * Emit timestamps shifted onto the current wall clock, preserving every
     * interval exactly. LatencyProbe measures latency as
     * host_now_ns() - timestamp_ns, so without this a recording made last week
     * reports a week of latency and every latency number becomes noise. Off
     * pushes the file's own stamps through verbatim -- which are UTC, because
     * the pipeline rebased them before they were written.
     */
    bool rebase_timestamps = true;

    /**
     * How long a frame waits for a free slot before it is given up on. Drops
     * here mean the pipeline is not keeping up with the recorded rate, which is
     * a finding rather than something to paper over by blocking.
     */
    uint64_t slot_wait_ms = 500;
  };

  /**
   * @brief Open the file and resolve the topic, so geometry() is readable
   * immediately.
   *
   * The geometry comes from the first message on the chosen topic -- a
   * sensor_msgs/Image carries its own width, height, step and encoding, so
   * nothing outside the file has to be told what is in it.
   *
   * @throws std::runtime_error if the file cannot be opened, holds no image
   *         topic, does not have the one asked for (the message names what it
   *         does have), or its first message will not decode.
   */
  explicit McapReplaySource(const Config& config);
  ~McapReplaySource() override;

  McapReplaySource(const McapReplaySource&) = delete;
  McapReplaySource& operator=(const McapReplaySource&) = delete;

  const CameraGeometry& geometry() const override { return geometry_; }

  /** One in flight, one to fill. */
  uint32_t min_slot_count() const override { return 2; }

  void start(FrameSink& sink) override;
  void stop() override;

  bool finished() const override { return finished_.load(std::memory_order_acquire); }
  bool failed() const override { return failed_.load(std::memory_order_acquire); }
  const std::string& failure() const override { return failure_; }

  void set_finished_callback(std::function<void()> cb) override { on_finished_ = std::move(cb); }

  uint64_t delivered() const override { return delivered_.load(std::memory_order_relaxed); }

  /** "late=12 slot_drops=0 undecodable=0 loops=3" */
  std::string counters() const override;
  std::string notes() const override;

  /** What the file says about the clock it was recorded against. Provenance, and labelled so. */
  std::string ptp_status() override;

  /** Frames that had to wait for a slot. Non-zero means the pipeline is slower than the file. */
  uint64_t late() const { return late_.load(std::memory_order_relaxed); }

  /** Frames given up on because no slot came free within slot_wait_ms. */
  uint64_t slot_drops() const { return slot_drops_.load(std::memory_order_relaxed); }

  /** Messages on the topic that would not decode as a sensor_msgs/Image. Should be zero. */
  uint64_t undecodable() const { return undecodable_.load(std::memory_order_relaxed); }

  uint64_t loops() const { return loops_.load(std::memory_order_relaxed); }

  /** The topic being replayed, resolved from role/topic. */
  const std::string& topic() const { return topic_; }

  /** The file's frame_id for this topic, e.g. "left_optical". Empty until a frame is read. */
  const std::string& frame_id() const { return frame_id_; }

  /** Messages on the topic, from the summary. 0 when the file carries no statistics. */
  uint64_t message_count() const { return message_count_; }

 private:
  struct Impl;

  void run(FrameSink& sink);
  void finish(std::string failure_reason);

  // Hand back every slot the reader has finished with, then take a free one;
  // FrameSink::kNoSlot if none came free in time. consumed() is level-triggered,
  // so polling it is the contract rather than a workaround.
  void reclaim(FrameSink& sink);
  uint32_t acquire_slot(FrameSink& sink, bool& waited);

  Config config_;
  std::unique_ptr<Impl> impl_;
  std::string topic_;
  std::string frame_id_;
  uint64_t message_count_ = 0;
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
  std::atomic<uint64_t> undecodable_{0};
  std::atomic<uint64_t> loops_{0};
};

}  // namespace perception
