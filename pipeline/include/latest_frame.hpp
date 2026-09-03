#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "host_frame.hpp"

namespace perception {

/**
 * @brief Turns a push sink into a pull, latest-wins consumer -- without copying
 * the frame.
 *
 * DownloadStage hands every sink the same shared_ptr, on its own thread, once
 * per frame and in order. That is the right shape for a consumer that wants
 * every frame and can keep up (a recorder), and the wrong one for a consumer
 * that has its own loop, or that is sometimes slow and would rather have the
 * newest frame than a backlog. This is the adapter for the second kind:
 * register sink() with the stage, call acquire_latest() from the consumer's own
 * thread.
 *
 * Nothing here allocates or copies pixels. The frame stays in the producer's
 * pinned pool slot and this holds a reference to it, which is the whole reason
 * to reach for this rather than publishing into a HostFrameRing -- downstream
 * of a D2H there is no DMA target to starve, so the copy the tap ring makes on
 * the ingress side would buy nothing here.
 *
 * The cost, and it is the only one: a held slot is a slot the producer cannot
 * refill. This holder pins at most one frame, and the consumer pins the one it
 * is working on, so budget two pool slots per LatestFrame on top of the
 * producer's own in-flight depth. Run short and the producer drops -- at
 * DownloadStage::enqueue(), counted by its pool's drops().
 *
 * Deliberately not tied to DownloadStage, and deliberately free of
 * <cuda_runtime.h>: anything that produces a shared_ptr<const HostFrame> can
 * feed one of these.
 */
class LatestFrame {
 public:
  using Frame = std::shared_ptr<const HostFrame>;
  using Sink = std::function<void(const Frame&)>;

  /** @param name Appears in health_line(). */
  explicit LatestFrame(std::string name);

  LatestFrame(const LatestFrame&) = delete;
  LatestFrame& operator=(const LatestFrame&) = delete;

  /**
   * @brief The callable to register with the producer, e.g.
   *        `stage.add_sink(latest.sink())`.
   *
   * Stores and returns: it takes the holder's lock for the length of a pointer
   * assignment and never blocks, so it does not hold up the producer's thread
   * or the sinks queued behind it. This must outlive the producer, because the
   * returned function refers to it.
   */
  Sink sink();

  /**
   * @brief Take the newest frame, blocking until there is one.
   *
   * @return The frame, held until the caller drops it. Null once stop() has
   *         been called -- including when a frame was waiting, so that the exit
   *         is the same every time -- which is how a consumer thread exits.
   */
  Frame acquire_latest();

  /**
   * @brief The newest frame if one is waiting, null if not. Never blocks.
   *
   * For a consumer that already has a loop of its own -- a GL thread pacing on
   * the display -- where blocking on a frame is not an option.
   */
  Frame take_latest();

  /** @brief Wake the waiter and release the held frame. Idempotent. */
  void stop();

  const std::string& name() const { return name_; }

  /** Frames the producer offered this consumer. */
  uint64_t offered() const;

  /** Frames overwritten before the consumer looked. Not a fault; the point. */
  uint64_t skipped() const;

  /** "viewer: 780 offered, 41 skipped, holding 1" */
  std::string health_line() const;

 private:
  const std::string name_;

  mutable std::mutex mutex_;
  std::condition_variable frame_ready_;

  // At most one, so "newer than the last one handed out" is just "not null":
  // acquire_latest() moves it out, and the sink counts a skip when it displaces
  // one nobody took.
  Frame latest_;
  bool running_ = true;

  uint64_t offered_ = 0;
  uint64_t skipped_ = 0;
};

}  // namespace perception
