#include "ring_pair_consumer.hpp"

#include <cstdio>
#include <stdexcept>
#include "cuda_util.hpp"

namespace perception {
namespace {

constexpr std::chrono::microseconds kPollInterval{2000};

}  // namespace

RingPairConsumer::RingPairConsumer(DeviceRingBuffer& reference, DeviceRingBuffer& other, Config config,
                               uint32_t consumer_id, int device_id)
    : reference_(&reference),
      other_(&other),
      config_(config),
      consumer_id_(consumer_id),
      device_id_(device_id) {
  if (consumer_id_ >= reference_->max_consumers() || consumer_id_ >= other_->max_consumers()) {
    throw std::runtime_error("RingPairConsumer: consumer id is outside a ring's max_consumers");
  }
  if (reference_ == other_) {
    throw std::runtime_error("RingPairConsumer: the two rings must be different");
  }
}

RingPairConsumer::~RingPairConsumer() { stop(); }

void RingPairConsumer::start() {
  if (running_.exchange(true)) return;
  worker_ = std::thread([this] { run(); });
}

void RingPairConsumer::stop() {
  if (!running_.exchange(false)) return;
  reference_->wake_all();
  other_->wake_all();
  if (worker_.joinable()) worker_.join();
}

bool RingPairConsumer::step(cudaStream_t stream) {
  FramePeek peek;
  if (!reference_->view_latest_inplace(peek)) return false;
  if (have_last_ && peek.slot == last_slot_ && peek.slot_seq == last_seq_) return false;

  ReadLease reference = reference_->lease_latest(consumer_id_, stream);
  if (!reference.valid()) {
    reference_missed_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  const uint64_t published = reference_->published();
  if (have_last_ && published > last_published_) {
    reference_skipped_.fetch_add(published - last_published_ - 1, std::memory_order_relaxed);
  }
  last_published_ = published;

  last_slot_ = reference.slot();
  last_seq_ = reference.seq();
  have_last_ = true;

  ReadLease other = other_->lease_by_timestamp(reference.timestamp_ns(), config_.tolerance_ns,
                                               consumer_id_, stream, /*closest_match=*/true);

  uint32_t attempt = 0;
  for (; !other.valid() && attempt < config_.retry_attempts; ++attempt) {
    std::this_thread::sleep_for(config_.retry_wait);
    other = other_->lease_by_timestamp(reference.timestamp_ns(), config_.tolerance_ns,
                                       consumer_id_, stream, /*closest_match=*/true);
  }

  if (!other.valid()) {
    unpaired_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (attempt > 0) late_partner_.fetch_add(1, std::memory_order_relaxed);

  cuda_error_check(cudaStreamWaitEvent(stream, reference.data_ready_event(), 0),
                   "RingPairConsumer: cudaStreamWaitEvent(reference)");
  cuda_error_check(cudaStreamWaitEvent(stream, other.data_ready_event(), 0),
                   "RingPairConsumer: cudaStreamWaitEvent(other)");

  const int64_t skew_ns = static_cast<int64_t>(other.timestamp_ns()) -
                          static_cast<int64_t>(reference.timestamp_ns());
  const int64_t abs_skew = skew_ns < 0 ? -skew_ns : skew_ns;
  if (abs_skew > max_abs_skew_ns_.load(std::memory_order_relaxed)) {
    max_abs_skew_ns_.store(abs_skew, std::memory_order_relaxed);
  }

  const uint64_t pair_id = next_pair_id_++;
  paired_.fetch_add(1, std::memory_order_relaxed);

  if (on_pair_) on_pair_(reference, other, skew_ns, pair_id, stream);

  return true;
}

std::string RingPairConsumer::health_line() const {
  char buffer[224];  // six counters at full width, plus the skew
  std::snprintf(buffer, sizeof(buffer),
                "pair paired=%llu unpaired=%llu late=%llu missed=%llu skipped=%llu "
                "max_skew=%.1fus",
                static_cast<unsigned long long>(paired()),
                static_cast<unsigned long long>(unpaired()),
                static_cast<unsigned long long>(late_partner()),
                static_cast<unsigned long long>(reference_missed()),
                static_cast<unsigned long long>(reference_skipped()),
                static_cast<double>(max_abs_skew_ns()) * 1e-3);
  return buffer;
}

void RingPairConsumer::run() {
  if (cudaSetDevice(device_id_) != cudaSuccess) {
    std::fprintf(stderr, "ring_pair_consumer: cudaSetDevice failed on the worker thread\n");
    running_.store(false, std::memory_order_relaxed);
    return;
  }

  CudaStream stream;
  while (running_.load(std::memory_order_relaxed)) {
    try {
      if (!step(stream)) std::this_thread::sleep_for(kPollInterval);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "ring_pair_consumer: %s\n", e.what());
    }
  }
  cudaStreamSynchronize(stream);
}

}  // namespace perception
