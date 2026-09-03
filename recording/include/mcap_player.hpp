#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace perception {

/**
 * @brief One message off the file, as the player hands it out.
 *
 * `data` points into the reader's own buffer and is valid only for the
 * duration of the on_message() call
 */
struct ReplayMessage {
  std::string_view topic;
  std::string_view schema_name;       ///< "sensor_msgs/msg/Imu"; empty for schema id 0.
  std::string_view message_encoding;  ///< "cdr".

  uint64_t log_time_ns = 0;  ///< As recorded. What pacing is measured against.

  /**
   * What to publish this as: the recorded stamp mapped onto the current wall
   * clock, or the recorded stamp verbatim when rebasing is off.
   */
  uint64_t stamp_ns = 0;

  uint32_t sequence = 0;
  const std::byte* data = nullptr;
  std::size_t size = 0;
};

/** @brief Anything that wants one topic's messages. */
class ReplaySubscriber {
 public:
  virtual ~ReplaySubscriber() = default;

  /**
   * @brief Called on the player's thread, in recorded order.
   * @return False if the message was dropped, which the player counts against
   *         this topic. Returning false is a finding, not an error.
   */
  virtual bool on_message(const ReplayMessage&) = 0;
};

/**
 * @brief Replays an MCAP with every topic in the file,
 * on one clock, at the spacing it was recorded with.
 *
 * The cost of one thread is that a subscriber which blocks delays every other
 * topic. That is the accepted trade: pacing is absolute rather than
 * frame-to-frame, so a stall is caught up rather than accumulated, and a drop
 * is a finding rather than something to hide behind a queue.
 */
class McapPlayer {
 public:
  struct Config {
    /** The .mcap file to replay. */
    std::string path;

    /**
     * Topics to replay. EMPTY REPLAYS EVERY TOPIC IN THE FILE, which is the
     * point -- a recording should be replayable without first being described.
     * A name nothing in the file carries is an error, not silence.
     */
    std::vector<std::string> topics;

    /** Dropped after `topics` is applied. */
    std::vector<std::string> exclude;

    /**
     * Which image topics become pipeline streams, in order: image_topics[0] is
     * stream 0. Empty takes every sensor_msgs/Image topic in the file, sorted
     * by name, which puts "/left/..." before "/right/...".
     *
     * These are always replayed, whatever `topics` says: they are what the GPU
     * pipeline is being fed.
     */
    std::vector<std::string> image_topics;

    /**
     * Playback rate. Above 1.0 this is a load knob as much as a time one: it
     * raises the frame rate the pipeline has to keep up with.
     */
    double speed = 1.0;

    /** Restart at the end rather than finishing the run. */
    bool loop = true;

    /**
     * Emit timestamps mapped onto the current wall clock, preserving every
     * interval and every cross-topic relationship exactly. LatencyProbe
     * measures latency as host_now_ns() - timestamp_ns, so without this a
     * recording made last week reports a week of latency and every latency
     * number becomes noise. Off pushes the file's own stamps through verbatim
     * -- which are UTC, because the pipeline rebased them before they were
     * written.
     */
    bool rebase_timestamps = true;

    /** Skip this far into the recording before the first message. */
    double start_seconds = 0.0;

    /** How much to play from there; 0 runs to the end. */
    double duration_seconds = 0.0;

    /**
     * How long a subscriber may take to accept a message before it is given up
     * on. Only image subscribers wait at all -- it is how long a frame waits
     * for a free slot -- and drops here mean the pipeline is not keeping up
     * with the recorded rate, which is a finding rather than something to paper
     * over by blocking.
     */
    uint64_t slot_wait_ms = 500;
  };

  /** One selected channel of the file. */
  struct ChannelInfo {
    std::string topic;
    std::string schema_name;
    std::string message_encoding;
    uint64_t message_count = 0;  ///< From the summary; 0 if the file carries no statistics.
  };

  /**
   * @brief Open the file and resolve the selection, so channels() and
   * image_topics() are readable immediately.
   *
   * @throws std::runtime_error if the file cannot be opened, holds no channel,
   *         or `topics` / `image_topics` name something it does not have -- the
   *         message lists what it does.
   */
  explicit McapPlayer(const Config& config);
  ~McapPlayer();

  McapPlayer(const McapPlayer&) = delete;
  McapPlayer& operator=(const McapPlayer&) = delete;

  const std::string& path() const { return config_.path; }

  /** Every channel that will be replayed, in name order. */
  const std::vector<ChannelInfo>& channels() const { return channels_; }

  /** The resolved image topics, in stream order. */
  const std::vector<std::string>& image_topics() const { return image_topics_; }

  /** How long a subscriber may take before a message is given up on. */
  uint64_t slot_wait_ms() const { return config_.slot_wait_ms; }

  /**
   * @brief What the file says about the clock it was recorded against.
   *
   * Read out of the recorder's own metadata record. Provenance, not a live lock
   * state: there is no clock being disciplined here.
   */
  const std::string& recorded_clock() const { return recorded_clock_; }

  /**
   * @brief Read the first message on `topic`, before start().
   *
   * How an image subscriber learns its geometry without the player having to
   * know what an image is.
   *
   * @return False if the topic carries no message.
   * @throws std::runtime_error if `topic` is not being replayed.
   */
  bool read_first(std::string_view topic, const std::function<void(const ReplayMessage&)>& body);

  /**
   * @brief Route `topic` to `subscriber`.
   * @throws std::runtime_error if `topic` is not being replayed, or already has
   *         a subscriber. A topic with no subscriber is still read and counted.
   */
  void subscribe(std::string_view topic, ReplaySubscriber& subscriber);

  /** @brief The same, for a consumer that does not want to be a class. */
  void subscribe(std::string_view topic, std::function<void(const ReplayMessage&)> callback);

  /**
   * @brief Say that one more subscriber has to bind before playback may begin.
   *
   * An image subscriber does not have its sink until FrameSource::start(), and
   * the composing app starts its streams one at a time -- so the player counts
   * the binds it is waiting for and launches itself when the last one lands.
   * Without this the first stream to start would race the second past the
   * frames it should have seen.
   */
  void expect_bind();

  /** @brief One expected bind has happened; starts playback when none are left. */
  void bind_done();

  /** @brief Begin playback. Idempotent, and implied by the last bind_done(). */
  void start();

  /** @brief Stop and join. Idempotent, and safe to call from every subscriber. */
  void stop();

  /**
   * @brief False once stop() has been asked for.
   *
   * A subscriber that waits -- an image sink waiting for a free slot -- runs on
   * the player's own thread, so finished() cannot become true underneath it and
   * this is the only thing that can tell it to give up. Without it a stop costs
   * slot_wait_ms per waiting subscriber.
   */
  bool running() const { return running_.load(std::memory_order_relaxed); }

  bool finished() const { return finished_.load(std::memory_order_acquire); }
  bool failed() const { return failed_.load(std::memory_order_acquire); }
  const std::string& failure() const { return failure_; }

  /**
   * @brief Called once on the player's thread when no further message can
   * arrive. Set before start(); throws are swallowed.
   */
  void add_finished_callback(std::function<void()> callback);

  /** Messages handed to a subscriber that accepted them. */
  uint64_t delivered(std::string_view topic) const;

  /** Messages a subscriber refused, because it had nowhere to put them. */
  uint64_t dropped(std::string_view topic) const;

  /** Messages read on a topic nobody subscribed to. Not a fault. */
  uint64_t unrouted(std::string_view topic) const;

  uint64_t loops() const { return loops_.load(std::memory_order_relaxed); }

  /**
   * @brief True when the file carries no message indexes and is being read in
   * the order it was written rather than in log-time order.
   *
   * A recording whose run was killed before close() finalised it. It still
   * replays -- see the note in the constructor -- but a reader is entitled to
   * know the ordering guarantee is the writer's rather than the file's.
   */
  bool unindexed() const { return unindexed_; }

  /**
   * @brief What the mcap reader complained about, if anything.
   *
   * The reader reports trouble through a callback rather than a throw, so
   * swallowing it turns "this file cannot be read that way" into "this file is
   * empty". Kept and reported instead.
   */
  const std::string& problem() const { return problem_; }

  /** "loops=3 dropped=0 unrouted=412" over every topic. */
  std::string counters() const;

  /** One line per replayed topic: what it is and how it is going. */
  std::string health_line() const;

 private:
  struct Impl;

  /** One selected channel's routing and counters. */
  struct Route {
    ChannelInfo info;
    ReplaySubscriber* subscriber = nullptr;

    // Owned when subscribe() was handed a callback rather than a class.
    std::unique_ptr<ReplaySubscriber> owned;

    std::atomic<uint64_t> delivered{0};
    std::atomic<uint64_t> dropped{0};
    std::atomic<uint64_t> unrouted{0};
  };

  void run();
  void finish(std::string failure_reason);

  // Keeps the first complaint; the rest are almost always the same one again.
  void note_problem(const std::string& what);
  Route& route(std::string_view topic);
  const Route& route(std::string_view topic) const;

  Config config_;
  std::unique_ptr<Impl> impl_;

  std::vector<ChannelInfo> channels_;
  std::vector<std::string> image_topics_;
  std::string recorded_clock_;
  std::string problem_;
  bool unindexed_ = false;

  // By pointer because of the atomics, and a map because routing is a lookup
  // per message and the set is tiny and fixed after construction.
  std::map<std::string, std::unique_ptr<Route>, std::less<>> routes_;

  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<uint32_t> pending_binds_{0};
  bool started_ = false;

  std::string failure_;
  std::atomic<bool> failed_{false};
  std::atomic<bool> finished_{false};
  std::vector<std::function<void()>> on_finished_;

  std::atomic<uint64_t> loops_{0};
};

}  // namespace perception
