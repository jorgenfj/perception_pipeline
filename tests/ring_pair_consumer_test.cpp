// Behaviour tests for live stereo pairing. Needs a GPU, no camera.
#include "ring_pair_consumer.hpp"

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "cuda_util.hpp"
#include "device_ring_buffer.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++failures;
}

using perception::CudaStream;
using perception::DeviceRingBuffer;
using perception::ReadLease;
using perception::RingPairConsumer;
using perception::WriteLease;
using perception::WritePolicy;

constexpr uint64_t kPeriod = 16'666'667;
constexpr uint64_t kTol = 500'000;
constexpr std::size_t kSlotBytes = 256;
constexpr uint32_t kConsumerId = 0;

DeviceRingBuffer make_ring(uint32_t depth = 4) {
  return DeviceRingBuffer(depth, kSlotBytes, perception::ReuseWait::DeviceWait,
                          WritePolicy::ScanForFree, /*max_consumers=*/2, /*device_id=*/0);
}

// Publish one frame at `timestamp_ns`.
void publish(DeviceRingBuffer& ring, cudaStream_t stream, uint64_t timestamp_ns) {
  WriteLease lease = ring.acquire_write(stream);
  if (!lease.valid()) {
    check(false, "test setup: could not acquire a write slot");
    return;
  }
  lease.publish(timestamp_ns);
}

RingPairConsumer::Config config(uint32_t retries = 0) {
  RingPairConsumer::Config c;
  c.tolerance_ns = kTol;
  c.retry_attempts = retries;
  c.retry_wait = std::chrono::milliseconds(1);
  return c;
}

void test_pairs_when_both_present() {
  DeviceRingBuffer a = make_ring();
  DeviceRingBuffer b = make_ring();
  CudaStream stream;

  RingPairConsumer stereo(a, b, config(), kConsumerId, 0);

  int64_t seen_skew = 0;
  uint64_t seen_pair_id = ~0ull;
  stereo.set_pair_callback([&](const ReadLease& ref, const ReadLease& other, int64_t skew,
                              uint64_t pair_id, cudaStream_t) {
    seen_skew = skew;
    seen_pair_id = pair_id;
    check(ref.valid() && other.valid(), "the callback gets two valid leases");
  });

  publish(b, stream, kPeriod + 300'000);  // partner first, 300us late
  publish(a, stream, kPeriod);

  check(stereo.step(stream), "a reference frame with a partner pairs");
  check(stereo.paired() == 1 && stereo.unpaired() == 0, "the pair is counted");
  check(seen_skew == 300'000, "the callback receives the signed skew");
  check(seen_pair_id == 0, "pair ids start at zero");
  check(stereo.max_abs_skew_ns() == 300'000, "max skew is tracked");
  check(stereo.late_partner() == 0, "a partner already present is not counted as late");

  check(!stereo.step(stream), "the same reference frame is not paired twice");
  check(stereo.paired() == 1, "and the count does not move");
}

void test_unpaired_when_partner_absent() {
  DeviceRingBuffer a = make_ring();
  DeviceRingBuffer b = make_ring();
  CudaStream stream;

  RingPairConsumer stereo(a, b, config(), kConsumerId, 0);

  publish(a, stream, kPeriod);
  check(!stereo.step(stream), "a reference frame with no partner does not pair");
  check(stereo.unpaired() == 1 && stereo.paired() == 0, "it is counted unpaired");

  // A partner outside the tolerance is no partner at all.
  publish(b, stream, 2 * kPeriod + kTol + 1);
  publish(a, stream, 2 * kPeriod);
  check(!stereo.step(stream), "a partner one nanosecond outside the tolerance does not pair");
  check(stereo.unpaired() == 2, "and is counted unpaired");

  // Exactly at the tolerance does pair, matching the offline merge.
  publish(b, stream, 3 * kPeriod + kTol);
  publish(a, stream, 3 * kPeriod);
  check(stereo.step(stream), "a partner exactly at the tolerance pairs");
  check(stereo.paired() == 1, "and is counted paired");
}

void test_moves_on_after_an_unpaired_frame() {
  DeviceRingBuffer a = make_ring();
  DeviceRingBuffer b = make_ring();
  CudaStream stream;

  RingPairConsumer stereo(a, b, config(), kConsumerId, 0);

  // An unpairable frame must not wedge the consumer: if the bookkeeping only
  // advanced on success, this frame would be retried forever and the next one
  // would never be looked at.
  publish(a, stream, kPeriod);
  check(!stereo.step(stream), "the unpairable frame does not pair");
  check(!stereo.step(stream), "and is not retried");
  check(stereo.unpaired() == 1, "so it is only counted once");

  publish(b, stream, 2 * kPeriod);
  publish(a, stream, 2 * kPeriod);
  check(stereo.step(stream), "the next frame is still reached");
  check(stereo.paired() == 1, "and pairs normally");
}

void test_late_partner_retry() {
  DeviceRingBuffer a = make_ring();
  DeviceRingBuffer b = make_ring();
  CudaStream stream;

  RingPairConsumer stereo(a, b, config(/*retries=*/5), kConsumerId, 0);

  publish(a, stream, kPeriod);

  // The partner lands while step() is in its retry wait.
  std::thread producer([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CudaStream own;
    publish(b, own, kPeriod);
    cudaStreamSynchronize(own);
  });

  const bool paired = stereo.step(stream);
  producer.join();

  check(paired, "a partner that lands during the retry still pairs");
  check(stereo.paired() == 1 && stereo.unpaired() == 0, "and is not counted unpaired");
  check(stereo.late_partner() == 1, "the retry is recorded as a late partner");
}

void test_counts_skipped_reference_frames() {
  DeviceRingBuffer a = make_ring();
  DeviceRingBuffer b = make_ring();
  CudaStream stream;

  RingPairConsumer stereo(a, b, config(), kConsumerId, 0);

  // One pair, stepped straight away: nothing had a chance to go past.
  publish(b, stream, kPeriod);
  publish(a, stream, kPeriod);
  check(stereo.step(stream), "the first reference frame pairs");
  check(stereo.reference_skipped() == 0, "and nothing is skipped on the first step");

  // Three more reference frames while the consumer is not looking. step()
  // takes `latest`, so the middle two are never attempted.
  for (uint64_t k = 2; k <= 4; ++k) {
    publish(b, stream, k * kPeriod);
    publish(a, stream, k * kPeriod);
  }

  check(stereo.step(stream), "the newest reference frame pairs");
  check(stereo.reference_skipped() == 2, "the two frames stepped over are counted skipped");
  check(stereo.paired() == 2, "they are not counted as pairs");
  check(stereo.unpaired() == 0, "nor as unpaired -- they were never looked up");

  // A step with no new reference frame leaves it alone.
  check(!stereo.step(stream), "no new reference frame is not a step");
  check(stereo.reference_skipped() == 2, "and skips nothing");
}

void test_rejects_bad_construction() {
  DeviceRingBuffer a = make_ring();
  DeviceRingBuffer b = make_ring();

  auto throws = [&](RingPairConsumer::Config c, uint32_t id) {
    try {
      RingPairConsumer stereo(a, b, c, id, 0);
    } catch (const std::exception&) {
      return true;
    }
    return false;
  };

  check(throws(config(), /*id=*/9), "a consumer id outside max_consumers is refused");

  bool same_ring_threw = false;
  try {
    RingPairConsumer stereo(a, a, config(), kConsumerId, 0);
  } catch (const std::exception&) {
    same_ring_threw = true;
  }
  check(same_ring_threw, "pairing a ring with itself is refused");
}

void test_threaded() {
  DeviceRingBuffer a = make_ring(6);
  DeviceRingBuffer b = make_ring(6);
  CudaStream stream;

  RingPairConsumer stereo(a, b, config(/*retries=*/2), kConsumerId, 0);
  stereo.start();

  constexpr int kFrames = 8;
  for (int i = 1; i <= kFrames; ++i) {
    publish(b, stream, static_cast<uint64_t>(i) * kPeriod + 100'000);
    publish(a, stream, static_cast<uint64_t>(i) * kPeriod);
    cudaStreamSynchronize(stream);
    std::this_thread::sleep_for(std::chrono::milliseconds(6));
  }

  // The consumer only ever sees the newest frame, so it will not have paired
  // all of them; what matters is that it paired and lost nothing to a fault.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  stereo.stop();

  check(stereo.paired() > 0, "the worker thread pairs frames");
  check(stereo.unpaired() == 0, "and none of them go unpaired");
  check(stereo.max_abs_skew_ns() == 100'000, "the skew is what was published");
  check(!stereo.running(), "stop() leaves it stopped");

  stereo.stop();
  check(true, "stop() is idempotent");
}

}  // namespace

int main() {
  if (cudaSetDevice(0) != cudaSuccess) {
    std::printf("FAILED: no CUDA device\n");
    return 1;
  }

  test_pairs_when_both_present();
  test_unpaired_when_partner_absent();
  test_moves_on_after_an_unpaired_frame();
  test_late_partner_retry();
  test_counts_skipped_reference_frames();
  test_rejects_bad_construction();
  test_threaded();

  std::printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
