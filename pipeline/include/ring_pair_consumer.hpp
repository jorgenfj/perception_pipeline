#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

#include "cuda_util.hpp"
#include "device_ring_buffer.hpp"
#include "types.hpp"

namespace perception {

// Pairs two device rings by timestamp: lease the newest frame from `reference`,
// find the frame in `other` whose stamp is within `config.tolerance_ns`, and
// hand both leases to the callback.
class RingPairConsumer {
 public:
  struct Config {
    uint64_t tolerance_ns = 500'000;

    // Retry for a partner still in flight
    uint32_t retry_attempts = 1;
    std::chrono::milliseconds retry_wait{2};
  };

  // Invoked on the consumer thread with both leases still held, and with
  // `stream` already ordered behind both frames' data-ready events -- so work
  // enqueued on it may read either frame directly. Do not block in here: the
  // leases are holding a slot in each ring.
  using PairCallback = std::function<void(const ReadLease& reference, const ReadLease& other,
                                          int64_t skew_ns, uint64_t pair_id, cudaStream_t stream)>;

  RingPairConsumer(DeviceRingBuffer& reference, DeviceRingBuffer& other, Config config,
                 uint32_t consumer_id, int device_id);
  ~RingPairConsumer();

  RingPairConsumer(const RingPairConsumer&) = delete;
  RingPairConsumer& operator=(const RingPairConsumer&) = delete;

  // Must be set before start().
  void set_pair_callback(PairCallback callback) { on_pair_ = std::move(callback); }

  void start();
  void stop();
  bool running() const { return running_.load(std::memory_order_relaxed); }

  // One reference frame through the lookup. Public so the pairing can be
  // driven inline by a caller that already owns a thread and a stream -- do not
  // mix that with start(), since the last-frame bookkeeping is single-consumer.
  // False when there is no new reference frame, or it could not be paired.
  bool step(cudaStream_t stream);

  // Pairs emitted.
  uint64_t paired() const { return paired_.load(std::memory_order_relaxed); }

  // Reference frames with no partner inside the tolerance, after the retries.
  // This is the sync number: PTP, trigger, or one camera slow or stopped.
  uint64_t unpaired() const { return unpaired_.load(std::memory_order_relaxed); }

  // Pairs that only came together on a retry -- the partner was in flight.
  // Harmless in ones and twos; climbing means transport jitter is eating into
  // the frame period.
  uint64_t late_partner() const { return late_partner_.load(std::memory_order_relaxed); }

  // Reference frames skipped because lease_latest() lost the slot to the
  // producer between the peek and the lease. Not a pairing failure.
  uint64_t reference_missed() const { return reference_missed_.load(std::memory_order_relaxed); }

  // Reference frames skipped between step calls
  uint64_t reference_skipped() const { return reference_skipped_.load(std::memory_order_relaxed); }

  // Worst |skew| in any pair so far, nanoseconds. With PTP locked and Scheduled
  // Action Commands armed this should stay in the microseconds.
  int64_t max_abs_skew_ns() const { return max_abs_skew_ns_.load(std::memory_order_relaxed); }

  // e.g. "pair paired=1204 unpaired=0 late=3 max_skew=41us"
  std::string health_line() const;

 private:
  void run();

  DeviceRingBuffer* reference_;
  DeviceRingBuffer* other_;
  Config config_;
  uint32_t consumer_id_;
  int device_id_;

  PairCallback on_pair_;

  uint32_t last_slot_ = 0;
  uint64_t last_seq_ = 0;
  uint64_t last_published_ = 0;
  bool have_last_ = false;
  uint64_t next_pair_id_ = 0;

  std::thread worker_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> paired_{0};
  std::atomic<uint64_t> unpaired_{0};
  std::atomic<uint64_t> late_partner_{0};
  std::atomic<uint64_t> reference_missed_{0};
  std::atomic<uint64_t> reference_skipped_{0};
  std::atomic<int64_t> max_abs_skew_ns_{0};
};

}  // namespace perception
