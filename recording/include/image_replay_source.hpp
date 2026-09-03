#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "frame_source.hpp"
#include "mcap_player.hpp"

namespace perception {

/**
 * @brief One image topic of a replayed MCAP, delivered into a FrameSink.
 */
class ImageReplaySource final : public FrameSource, public ReplaySubscriber {
 public:
  /**
   * @brief Bind to one of the player's image topics.
   *
   * Reads the first message on the topic, so geometry() is readable
   * immediately
   *
   * @throws std::runtime_error if the topic is not being replayed, carries no
   *         message, its first message will not decode, or it is encoded in
   *         something this pipeline has no pixel format for.
   */
  ImageReplaySource(McapPlayer& player, std::string topic);
  ~ImageReplaySource() override;

  ImageReplaySource(const ImageReplaySource&) = delete;
  ImageReplaySource& operator=(const ImageReplaySource&) = delete;

  const CameraGeometry& geometry() const override { return geometry_; }

  /** One in flight, one to fill. */
  uint32_t min_slot_count() const override { return 2; }

  /**
   * @brief Bind the sink and tell the player one more source is ready.
   *
   * Does not start a thread: the player owns the only one, and begins when the
   * last source has bound.
   */
  void start(FrameSink& sink) override;
  void stop() override;

  bool finished() const override { return player_.finished(); }
  bool failed() const override { return player_.failed(); }
  const std::string& failure() const override { return player_.failure(); }

  void set_finished_callback(std::function<void()> cb) override;

  uint64_t delivered() const override { return player_.delivered(topic_); }

  /** "late=12 slot_drops=0 undecodable=0 loops=3" */
  std::string counters() const override;
  std::string notes() const override;

  /** What the file says about the clock it was recorded against. Provenance, and labelled so. */
  std::string ptp_status() override;

  /** Frames that had to wait for a slot. Non-zero means the pipeline is slower than the file. */
  uint64_t late() const { return late_.load(std::memory_order_relaxed); }

  /** Frames given up on because no slot came free within the player's slot_wait_ms. */
  uint64_t slot_drops() const { return player_.dropped(topic_); }

  /** Messages on the topic that would not decode as a sensor_msgs/Image. Should be zero. */
  uint64_t undecodable() const { return undecodable_.load(std::memory_order_relaxed); }

  const std::string& topic() const { return topic_; }

  /** The file's frame_id for this topic, e.g. "left_optical". */
  const std::string& frame_id() const { return frame_id_; }

  /** Messages on the topic, from the summary. 0 when the file carries no statistics. */
  uint64_t message_count() const { return message_count_; }

 private:
  bool on_message(const ReplayMessage& message) override;

  // Hand back every slot the reader has finished with, then take a free one;
  // FrameSink::kNoSlot if none came free in time. consumed() is level-triggered,
  // so polling it is the contract rather than a workaround.
  void reclaim();
  uint32_t acquire_slot(bool& waited);

  McapPlayer& player_;
  std::string topic_;
  std::string frame_id_;
  uint64_t message_count_ = 0;
  uint64_t slot_wait_ms_ = 500;
  CameraGeometry geometry_;

  // Set by start() and only read on the player's thread afterwards.
  FrameSink* sink_ = nullptr;

  // Indexed by sink slot: true from commit() until consumed() says the read
  // retired. Only the player's thread touches it.
  std::vector<bool> held_;

  uint32_t next_frame_id_ = 0;

  std::atomic<uint64_t> late_{0};
  std::atomic<uint64_t> undecodable_{0};
};

}  // namespace perception
