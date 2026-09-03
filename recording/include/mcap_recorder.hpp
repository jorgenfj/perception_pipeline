#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cdr_writer.hpp"

namespace perception {

/**
 * @brief Writes an arbitrary set of typed topics to an MCAP file as ROS 2
 * messages.
 *
 * push() encodes and returns; a writer thread does the disk work, and a disk
 * that cannot keep up drops messages rather than throttling the pipeline that
 * feeds it. A recorder that backed up would change the timing of the run it is
 * recording.
 *
 * Knows nothing about message content: a topic is a name, a schema and a byte
 * budget. Message types live in ros_messages.hpp.
 *
 * Encoding happens on the PUSHING thread, so the producer gets its buffer back
 * before push() returns rather than when the disk catches up.
 *
 * One queue budget and one buffer pool per topic, so a fast topic cannot evict
 * a slow one. One pushing thread per topic -- not enforced, but the pool is
 * sized for it.
 */
class McapRecorder {
 public:
  /** Index into the recorder's topics, returned by add_topic(). */
  using TopicId = uint32_t;

  struct Config {
    /** Parent directory; a timestamped .mcap is created inside it. */
    std::string root = "recordings";

    bool compress = false;

    /**
     * Seconds of each topic to hold while the disk catches up. Per topic and
     * scaled by its rate, so one number suits a 400Hz sensor and a 3.5Hz camera
     * at once.
     */
    double buffer_seconds = 2.0;

    /** Cap on one topic's queue, applied after buffer_seconds. */
    uint32_t topic_memory_mb = 16;

    /** mcap chunk size. The 768KiB default is smaller than one camera frame. */
    uint32_t chunk_mb = 4;

    /**
     * How often the open chunk is closed even when not full. Only whole chunks
     * are readable, so this bounds what a crash costs. Page cache, not fsync.
     */
    double flush_seconds = 2.0;

    /**
     * Provenance only: what the pipeline already subtracted from the sensor
     * clock to put these stamps on CLOCK_REALTIME, written into the file's
     * metadata.
     */
    int64_t epoch_offset_ns = 0;
  };

  /** One channel's declaration. ros_messages.hpp fills these from a type's traits. */
  struct Topic {
    std::string name;      ///< "/imu/data"
    std::string type;      ///< "sensor_msgs/msg/Imu"
    std::string schema;    ///< ros2msg text, from ros_schemas.hpp
    double rate_hz = 0.0;  ///< Sizes the queue only; not enforced or recorded.

    /**
     * Largest message this topic will encode. Buffers are reserved to it, so an
     * encode that exceeds it reallocates on the pushing thread; grew() counts
     * that.
     */
    std::size_t max_message_bytes = 0;

    /** 0 derives it from rate_hz and Config::buffer_seconds. */
    uint32_t queue_depth = 0;
  };

  /**
   * @brief Create the directory and open the file.
   * @throws std::runtime_error if the file cannot be opened.
   */
  explicit McapRecorder(const Config& config);
  ~McapRecorder();

  McapRecorder(const McapRecorder&) = delete;
  McapRecorder& operator=(const McapRecorder&) = delete;

  /**
   * @brief Declare a topic and get the id to push to.
   *
   * Before start() only: this writes schema and channel records, and
   * mcap::McapWriter is not thread-safe. Topics of one `type` share a schema.
   *
   * @param topic Name, type, schema and byte budget.
   * @return The id push() takes.
   *
   * @throws std::runtime_error after start(), if max_message_bytes is zero, or
   *         if one type is declared twice with different schema text.
   */
  TopicId add_topic(const Topic& topic);

  /**
   * @brief Allocate every topic's buffers and start the writer thread.
   * @throws std::runtime_error if no topics were declared.
   */
  void start();

  /**
   * @brief Encode and enqueue, on the calling thread.
   *
   * @param topic An id from add_topic().
   * @param timestamp_ns Message stamp; also its mcap log time.
   * @param encode Called with a CdrWriter positioned after the encapsulation
   *        header, outside the lock. Must write the whole body and nothing
   *        else, and must not throw.
   * @return False if the message was dropped, which drops(topic) counts: the
   *         topic's budget was full, or the recorder is not running.
   */
  template <typename Encode>
  bool push(TopicId topic, uint64_t timestamp_ns, Encode&& encode);

  /** @brief Drain, join the writer and close the file. Idempotent. */
  void close();

  const std::string& path() const { return path_; }
  std::size_t topic_count() const { return topics_.size(); }

  uint64_t written() const;
  uint64_t drops() const;
  uint64_t bytes_written() const { return bytes_written_.load(std::memory_order_relaxed); }

  uint64_t written(TopicId topic) const;
  uint64_t drops(TopicId topic) const;

  /**
   * Encodes that outgrew their topic's reserved buffer and reallocated on the
   * pushing thread. Non-zero is a max_message_bytes bug, not a disk problem.
   */
  uint64_t grew() const { return grew_.load(std::memory_order_relaxed); }

  /** Encoded bytes queued. The backlog signal: rising means the disk is behind. */
  uint64_t bytes_in_flight() const;
  uint64_t bytes_in_flight_peak() const {
    return bytes_in_flight_peak_.load(std::memory_order_relaxed);
  }

  /** Worst single mcap write, microseconds -- the disk's tail latency. */
  uint64_t write_max_us() const { return write_max_us_.load(std::memory_order_relaxed); }

  /**
   * "mcap: 1520 msgs, drops=0/0/0, peak 412/800 /imu/data, 2.1MB queued
   *  (peak 18.4MB), write_max=812us, 41.2MB"
   */
  std::string health_line() const;

 private:
  /** One queued message. No type tag, so nothing downstream branches on one. */
  struct Message {
    TopicId topic = 0;
    uint64_t timestamp_ns = 0;
    CdrWriter cdr;
  };

  /** One channel's budget, buffers and counters. By pointer because of the atomics. */
  struct TopicState {
    Topic declared;
    uint32_t channel_id = 0;  ///< An mcap::ChannelId, kept untyped so mcap:: stays out of here.
    uint32_t sequence = 0;    ///< Writer thread only.
    uint32_t depth = 0;       ///< Resolved from rate_hz x buffer_seconds.
    uint32_t queued = 0;      ///< Guarded by mutex_.

    /**
     * depth + 1 buffers: the queue holds at most depth, and the extra one covers
     * the message in the writer's hand or being encoded. A pusher can only take
     * one when the budget has room, so there is one drop condition and not two.
     */
    std::vector<CdrWriter> spare;

    std::atomic<uint64_t> written{0};
    std::atomic<uint64_t> drops{0};
    std::atomic<uint32_t> peak{0};
  };

  struct Impl;  // the mcap writer and its schema table

  void run();
  bool take_spare(TopicId topic, CdrWriter& out);
  bool enqueue(TopicId topic, uint64_t timestamp_ns, CdrWriter&& cdr);
  void recycle(TopicId topic, CdrWriter&& cdr);

  Config config_;
  std::string path_;
  std::unique_ptr<Impl> impl_;

  // One mutex across all topics: the critical sections are a deque move, and at
  // these rates that is a duty cycle around 0.005%. Splitting it would cost the
  // single wait that lets the writer sleep until any topic has work.
  mutable std::mutex mutex_;
  std::condition_variable queued_;
  std::vector<std::unique_ptr<TopicState>> topics_;

  // Arrival order across all topics; the per-topic budget is what keeps them
  // from evicting each other, so the queue itself does not need splitting.
  std::deque<Message> queue_;

  uint64_t bytes_in_flight_ = 0;  // guarded by mutex_, so it cannot disagree with queue_

  bool started_ = false;
  bool running_ = false;
  bool closed_ = false;

  std::thread thread_;
  std::atomic<uint64_t> bytes_written_{0};
  std::atomic<uint64_t> bytes_in_flight_peak_{0};
  std::atomic<uint64_t> write_max_us_{0};
  std::atomic<uint64_t> grew_{0};
};

template <typename Encode>
bool McapRecorder::push(TopicId topic, uint64_t timestamp_ns, Encode&& encode) {
  CdrWriter cdr;
  if (!take_spare(topic, cdr)) return false;  // counted there

  // Outside the lock. The capacity check is the guard that says the reserve was
  // big enough; if it was not, this push just allocated on a producer's thread.
  const std::size_t reserved = cdr.capacity();
  encode(cdr);
  if (cdr.capacity() != reserved) grew_.fetch_add(1, std::memory_order_relaxed);

  return enqueue(topic, timestamp_ns, std::move(cdr));
}

}  // namespace perception
