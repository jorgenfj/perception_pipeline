#include "stereo_consumer.hpp"

#include <cstdio>
#include <stdexcept>

#include "frame_pairing.hpp"

namespace perception {
namespace {

constexpr std::chrono::microseconds kPollInterval{2000};

}  // namespace

StereoConsumer::StereoConsumer(DeviceRingBuffer& reference, DeviceRingBuffer& other, Config config,
                               uint32_t consumer_id, int device_id)
    : reference_(&reference),
      other_(&other),
      config_(config),
      consumer_id_(consumer_id),
      device_id_(device_id) {
  if (consumer_id_ >= reference_->max_consumers() || consumer_id_ >= other_->max_consumers()) {
    throw std::runtime_error("StereoConsumer: consumer id is outside a ring's max_consumers");
  }
  if (reference_ == other_) {
    throw std::runtime_error("StereoConsumer: the two rings must be different");
  }
  // The greedy lookup is only unambiguous below half a period; above it, two
  // different frames qualify. Checked here so a bad config fails at startup
  // rather than silently pairing the wrong exposures.
  if (config_.frame_period_ns != 0) {
    require_pair_tolerance(config_.tolerance_ns, config_.frame_period_ns);
  }
}

StereoConsumer::~StereoConsumer() { stop(); }

void StereoConsumer::start() {
  if (running_.exchange(true)) return;
  worker_ = std::thread([this] { run(); });
}

void StereoConsumer::stop() {
  if (!running_.exchange(false)) return;
  // Both rings can be parked in wait_for_publish; wake them so the worker sees
  // running_ == false rather than waiting out a frame that may never come.
  reference_->wake_all();
  other_->wake_all();
  if (worker_.joinable()) worker_.join();
}

bool StereoConsumer::step(cudaStream_t stream) {
  FramePeek peek;
  if (!reference_->view_latest_inplace(peek)) return false;
  if (have_last_ && peek.slot == last_slot_ && peek.slot_seq == last_seq_) return false;

  ReadLease reference = reference_->lease_latest(consumer_id_, stream);
  if (!reference.valid()) {
    // The producer recycled the slot between the peek and the lease. Nothing to
    // pair, and nothing to record against a frame we never held.
    reference_missed_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  // Advance the bookkeeping as soon as the frame is in hand. If this waited
  // until a pair succeeded, an unpairable frame would be retried forever and
  // the consumer would never move on.
  last_slot_ = reference.slot();
  last_seq_ = reference.seq();
  have_last_ = true;

  // Tolerance zero would be exact-identity, but the reference timestamp comes
  // from a different ring than the one being searched, so it is the *camera's*
  // pair that has to fall inside the tolerance, not a copy of the same value.
  ReadLease other = other_->lease_by_timestamp(reference.timestamp_ns(), config_.tolerance_ns,
                                               consumer_id_, stream, /*closest_match=*/true);

  uint32_t attempt = 0;
  for (; !other.valid() && attempt < config_.retry_attempts; ++attempt) {
    // The partner may simply not have landed yet. Sleeping here holds the
    // reference slot, which is why the wait is short and the count is bounded:
    // the whole retry budget is retry_attempts * retry_wait, and stop() waits
    // out at most that much. Deliberately not conditioned on running_ -- step()
    // is also callable inline, where running_ is false the whole time.
    std::this_thread::sleep_for(config_.retry_wait);
    other = other_->lease_by_timestamp(reference.timestamp_ns(), config_.tolerance_ns,
                                       consumer_id_, stream, /*closest_match=*/true);
  }

  if (!other.valid()) {
    unpaired_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (attempt > 0) late_partner_.fetch_add(1, std::memory_order_relaxed);

  // Order the caller's stream behind both frames' production before anything
  // reads either one.
  cuda_error_check(cudaStreamWaitEvent(stream, reference.data_ready_event(), 0),
                   "StereoConsumer: cudaStreamWaitEvent(reference)");
  cuda_error_check(cudaStreamWaitEvent(stream, other.data_ready_event(), 0),
                   "StereoConsumer: cudaStreamWaitEvent(other)");

  const int64_t skew_ns = static_cast<int64_t>(other.timestamp_ns()) -
                          static_cast<int64_t>(reference.timestamp_ns());
  const int64_t abs_skew = skew_ns < 0 ? -skew_ns : skew_ns;
  if (abs_skew > max_abs_skew_ns_.load(std::memory_order_relaxed)) {
    max_abs_skew_ns_.store(abs_skew, std::memory_order_relaxed);
  }

  const uint64_t pair_id = next_pair_id_++;
  paired_.fetch_add(1, std::memory_order_relaxed);

  if (on_pair_) on_pair_(reference, other, skew_ns, pair_id, stream);

  // Both leases drop here, recording read-completion against `stream`. The
  // reading work only has to be enqueued by now, not finished.
  return true;
}

std::string StereoConsumer::health_line() const {
  char buffer[160];
  std::snprintf(buffer, sizeof(buffer),
                "stereo paired=%llu unpaired=%llu late=%llu missed=%llu max_skew=%.1fus",
                static_cast<unsigned long long>(paired()),
                static_cast<unsigned long long>(unpaired()),
                static_cast<unsigned long long>(late_partner()),
                static_cast<unsigned long long>(reference_missed()),
                static_cast<double>(max_abs_skew_ns()) * 1e-3);
  return buffer;
}

void StereoConsumer::run() {
  if (cudaSetDevice(device_id_) != cudaSuccess) {
    std::fprintf(stderr, "stereo: cudaSetDevice failed on the worker thread\n");
    running_.store(false, std::memory_order_relaxed);
    return;
  }

  CudaStream stream;
  while (running_.load(std::memory_order_relaxed)) {
    try {
      if (!step(stream)) std::this_thread::sleep_for(kPollInterval);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "stereo: %s\n", e.what());
    }
  }
  cudaStreamSynchronize(stream);
}

}  // namespace perception
