#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "host_frame.hpp"

namespace perception {

/**
 * @brief One copy of each camera frame, shared by every CPU consumer that wants
 * it, with latest-wins reads.
 *
 * The tap between the camera and everything that is not the GPU. commit() on
 * the acquisition thread publishes one frame here -- a single memcpy however
 * many consumers there are -- and the ingress slot goes straight back to the
 * transport, whose slots are the camera's own DMA targets.
 *
 * Each consumer holds exactly one frame at a time, for as long as it likes. A
 * consumer that takes 200ms pins one slot and skips whatever arrived meanwhile;
 * it cannot slow the camera, and it cannot make another consumer miss anything.
 * That is the whole point: nobody downstream has to care how fast it releases.
 *
 * Latest-wins, and only that. A consumer always gets the newest frame and
 * counts what it stepped over, because the alternative -- reserving the frames
 * ahead of a lagging reader -- is a queue, and a queue here would make one slow
 * consumer everyone else's problem. What is skipped is counted per consumer, so
 * a gap is attributable rather than mysterious.
 *
 * No WritePolicy. DeviceRingBuffer offers RoundRobin because a CUDA graph bakes
 * the destination pointer and so needs a predictable slot index; there is no
 * graph here, and this producer must never block, which RoundRobin cannot
 * support -- see try_acquire_write() there, which refuses it outright.
 */
class HostFrameRing {
 public:
  static constexpr uint32_t kNoSlot = ~0u;

  /**
   * @brief Allocate the slots.
   *
   * @param slots Sized `consumers + 2`: one being written, one holding the
   *        newest published frame, and one per consumer.
   * @param frame_bytes The largest frame that will be published.
   *
   * @throws std::runtime_error if `slots` is under 2 or `frame_bytes` is zero.
   */
  HostFrameRing(uint32_t slots, std::size_t frame_bytes, uint32_t width, uint32_t height);
  ~HostFrameRing();

  HostFrameRing(const HostFrameRing&) = delete;
  HostFrameRing& operator=(const HostFrameRing&) = delete;

  /**
   * @brief Register a reader and get the id it acquires with.
   *
   * Before the first publish(): the cursor a consumer starts from has to be set
   * without racing the producer.
   *
   * @param name Appears in health_line().
   * @return The id acquire_latest() takes.
   * @throws std::runtime_error after publishing has begun.
   */
  uint32_t add_consumer(std::string name);

  /**
   * @brief Copy a frame in and make it the newest. Never blocks.
   *
   * @param bytes What was actually written, which the published frame reports
   *        as HostFrame::bytes -- the payload length, not the slot's capacity.
   *        A short frame must not be recorded or published as a full one. Must
   *        not exceed the frame_bytes given at construction.
   * @return False if every slot was pinned, which drops() counts. The frame is
   *         gone; there is nothing for the caller to do about it.
   */
  bool publish(const void* data, std::size_t bytes, uint64_t timestamp_ns);

  /**
   * @brief Take the newest frame, releasing whatever this consumer held.
   *
   * Blocks until a frame newer than the one this consumer last saw exists.
   *
   * @param consumer An id from add_consumer().
   * @return The frame, held until the last copy of this pointer goes -- so a
   *         consumer releases its previous frame by overwriting the pointer, and
   *         holds two for the moment in between. That moment is what the second
   *         spare slot in the sizing rule covers. Null once stop() has been
   *         called, which is how a consumer thread exits.
   */
  std::shared_ptr<const HostFrame> acquire_latest(uint32_t consumer);

  /** @brief Wake every waiter so it can exit. Idempotent. */
  void stop();

  uint32_t slots() const;

  /** Slot capacity, i.e. the largest frame publish() will take. */
  std::size_t frame_bytes() const;

  uint64_t published() const { return published_.load(std::memory_order_relaxed); }

  /** Frames the producer could not place because every slot was pinned. */
  uint64_t drops() const { return drops_.load(std::memory_order_relaxed); }

  /** Frames this consumer stepped over to reach the newest. Not a fault. */
  uint64_t skipped(uint32_t consumer) const;

  uint32_t in_use() const;
  uint32_t peak_in_use() const;

  /** "tap: 780 published, drops=0, held 2/4 (peak 3), skipped aruco=41 recorder=0" */
  std::string health_line() const;

 private:
  // Held by the frames' deleters as well as the ring, so a frame outliving the
  // ring is safe -- the same reason HostFramePool does it.
  struct Store;

  struct Consumer {
    std::string name;
    uint64_t cursor = 0;  // sequence of the last frame handed to this consumer
    std::atomic<uint64_t> skipped{0};
  };

  std::shared_ptr<Store> store_;
  std::vector<std::unique_ptr<Consumer>> consumers_;
  bool publishing_ = false;

  std::atomic<uint64_t> published_{0};
  std::atomic<uint64_t> drops_{0};
};

}  // namespace perception
