// Behaviour test for the host-to-device boundary. Needs a GPU but no camera:
// the "source" is this file, filling pinned slots with a known pattern.

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cuda_util.hpp"
#include "device_ring_buffer.hpp"
#include "device_transform.hpp"
#include "host_ingress_ring.hpp"
#include "transform_stage.hpp"
#include "types.hpp"
#include "upload_stage.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

using perception::DeviceRingBuffer;
using perception::HostIngressRing;
using perception::ImageDesc;
using perception::PassThroughTransform;
using perception::PixelFormat;
using perception::TransformStage;
using perception::UploadStage;

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 48;
constexpr uint32_t kIngressDepth = 4;
constexpr uint32_t kDeviceDepth = 8;

const ImageDesc kDesc = perception::packed_desc(kWidth, kHeight, PixelFormat::GRAY8);

// Every byte of frame n is (n + index) & 0xff, so a torn upload -- a mix of two
// frames -- shows up as a mismatch rather than passing by luck.
void fill(void* dst, std::size_t bytes, uint8_t frame) {
  auto* p = static_cast<uint8_t*>(dst);
  for (std::size_t i = 0; i < bytes; ++i) p[i] = static_cast<uint8_t>(frame + i);
}

// Pull the device ring's newest slot back to the host and compare.
bool latest_matches(const DeviceRingBuffer& ring, uint8_t frame, uint64_t timestamp_ns) {
  perception::FrameView view;
  if (!ring.view_latest_inplace(view)) return false;
  if (view.frame.timestamp_ns != timestamp_ns) return false;

  if (cudaEventSynchronize(view.data_ready_event) != cudaSuccess) return false;

  std::vector<uint8_t> got(kDesc.bytes());
  if (cudaMemcpy(got.data(), view.frame.image_ptr, got.size(), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return false;
  }

  std::vector<uint8_t> want(kDesc.bytes());
  fill(want.data(), want.size(), frame);
  return std::memcmp(got.data(), want.data(), got.size()) == 0 && ring.read_was_clean(view);
}

// One frame in, source side. Returns false if the ring stopped underneath us.
bool stage_frame(HostIngressRing& ring, uint8_t frame, uint64_t timestamp_ns) {
  const uint32_t slot = ring.acquire();
  if (slot == HostIngressRing::kNoSlot) return false;
  fill(ring.host_ptr(slot), kDesc.bytes(), frame);
  ring.commit(slot, timestamp_ns, kDesc.bytes());
  return true;
}

void test_inline_step() {
  std::printf("inline step(), no worker thread:\n");

  HostIngressRing ingress(kIngressDepth, kDesc.bytes());
  DeviceRingBuffer device(kDeviceDepth, kDesc.bytes());
  PassThroughTransform transform;
  UploadStage stage(ingress, device, transform, kDesc);

  check(stage.output_desc().bytes() == kDesc.bytes(), "transform sizes the output ring");
  check(stage.has_transform() && stage.scratch_slots() == 4, "a transform allocates scratch");

  bool all_matched = true;
  // More frames than either ring is deep, so every slot is reused several times
  // and the interlocks are actually exercised.
  for (uint8_t n = 1; n <= 20; ++n) {
    if (!stage_frame(ingress, n, 1000ull * n)) {
      all_matched = false;
      break;
    }
    if (!stage.step()) {
      all_matched = false;
      break;
    }
    if (!latest_matches(device, n, 1000ull * n)) {
      all_matched = false;
      std::printf("      mismatch at frame %u\n", n);
      break;
    }
  }
  check(all_matched, "20 frames arrive intact, laps both rings");
  check(stage.uploaded() == 20, "uploaded counter tracks");
  check(stage.failed() == 0, "no failures");
}

void test_no_transform() {
  std::printf("upload only, no transform:\n");

  HostIngressRing ingress(kIngressDepth, kDesc.bytes());
  DeviceRingBuffer device(kDeviceDepth, kDesc.bytes());
  UploadStage stage(ingress, device, kDesc);

  check(!stage.has_transform(), "no transform attached");
  check(stage.scratch_slots() == 0, "no scratch allocated");
  check(stage.output_desc().bytes() == kDesc.bytes(), "output geometry mirrors the input");

  // Same frame count and pattern as the transform path, so the two are directly
  // comparable: identical bytes must come out either way.
  bool all_matched = true;
  for (uint8_t n = 1; n <= 20; ++n) {
    if (!stage_frame(ingress, n, 1000ull * n)) {
      all_matched = false;
      break;
    }
    if (!stage.step()) {
      all_matched = false;
      break;
    }
    if (!latest_matches(device, n, 1000ull * n)) {
      all_matched = false;
      std::printf("      mismatch at frame %u\n", n);
      break;
    }
  }
  check(all_matched, "20 frames land in the output slot straight from pinned memory");
  check(stage.uploaded() == 20, "uploaded counter tracks");
  check(stage.failed() == 0, "no failures");

  // The pinned slot is still released against the stage's stream on this path,
  // so the producer's interlock has to hold without any scratch involved.
  check(ingress.committed() == ingress.consumed(), "every pinned slot was returned");

  ingress.stop();
}

void test_no_transform_worker() {
  std::printf("upload only, driven by the worker thread:\n");

  constexpr uint32_t kFrames = 200;

  HostIngressRing ingress(kIngressDepth, kDesc.bytes());
  DeviceRingBuffer device(kDeviceDepth, kDesc.bytes());
  UploadStage stage(ingress, device, kDesc);

  stage.start();
  std::thread source([&] {
    for (uint32_t n = 0; n < kFrames; ++n) {
      if (!stage_frame(ingress, static_cast<uint8_t>(n), 1000ull * (n + 1))) return;
    }
  });
  source.join();

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (stage.uploaded() < kFrames && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  stage.stop();
  ingress.stop();

  check(stage.uploaded() == kFrames, "every staged frame was uploaded, none dropped");
  check(stage.failed() == 0, "no failures");
  check(latest_matches(device, static_cast<uint8_t>(kFrames - 1), 1000ull * kFrames),
        "last frame is intact on the device");
}

void test_worker_thread() {
  std::printf("worker thread draining the queue:\n");

  constexpr uint32_t kFrames = 300;

  HostIngressRing ingress(kIngressDepth, kDesc.bytes());
  DeviceRingBuffer device(kDeviceDepth, kDesc.bytes());
  PassThroughTransform transform;
  UploadStage stage(ingress, device, transform, kDesc);

  stage.start();

  // The source thread never touches CUDA -- that is the point of the split.
  std::thread source([&] {
    for (uint32_t n = 0; n < kFrames; ++n) {
      if (!stage_frame(ingress, static_cast<uint8_t>(n), 1000ull * (n + 1))) return;
    }
  });
  source.join();

  // Drain: the producer is done, so wait for the consumer to catch up.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (stage.uploaded() < kFrames && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  stage.stop();
  ingress.stop();

  check(stage.uploaded() == kFrames, "every staged frame was uploaded, none dropped");
  check(stage.failed() == 0, "no failures");
  check(ingress.committed() == ingress.consumed(), "every pinned slot was returned");
  check(latest_matches(device, static_cast<uint8_t>(kFrames - 1), 1000ull * kFrames),
        "last frame is intact on the device");
}

void test_backpressure() {
  std::printf("producer backpressure when nothing consumes:\n");

  HostIngressRing ingress(kIngressDepth, kDesc.bytes());
  DeviceRingBuffer device(kDeviceDepth, kDesc.bytes());
  PassThroughTransform transform;
  UploadStage stage(ingress, device, transform, kDesc);

  // Fill the ring without stepping, so nothing is ever released.
  for (uint32_t n = 0; n < kIngressDepth; ++n) {
    stage_frame(ingress, static_cast<uint8_t>(n), 1000ull * (n + 1));
  }

  std::atomic<bool> returned{false};
  std::thread producer([&] {
    stage_frame(ingress, 99, 99000);
    returned.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  check(!returned.load(), "acquire() blocks once the ring is full");

  // One step frees exactly one slot, which is all the blocked producer needs.
  check(stage.step(), "step() consumes the oldest staged frame");
  producer.join();
  check(returned.load(), "releasing a slot unblocks the producer");

  ingress.stop();
}

void test_stop_unblocks_producer() {
  std::printf("stop() releases a blocked producer:\n");

  HostIngressRing ingress(kIngressDepth, kDesc.bytes());
  for (uint32_t n = 0; n < kIngressDepth; ++n) {
    stage_frame(ingress, static_cast<uint8_t>(n), 1000ull * (n + 1));
  }

  std::atomic<uint32_t> got{0};
  std::thread producer([&] { got.store(ingress.acquire()); });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ingress.stop();
  producer.join();

  check(got.load() == HostIngressRing::kNoSlot, "acquire() returns kNoSlot after stop");
}

void test_abandon() {
  std::printf("abandoning an acquired slot:\n");

  HostIngressRing ingress(kIngressDepth, kDesc.bytes());

  const uint32_t first = ingress.acquire();
  ingress.abandon(first);
  const uint32_t again = ingress.acquire();
  check(first == again, "an abandoned slot comes straight back out");

  ingress.commit(again, 1, kDesc.bytes());
  check(ingress.committed() == 1, "an abandoned slot never reaches the consumer");

  bool threw = false;
  try {
    ingress.abandon(again);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "a committed slot cannot be abandoned");

  ingress.stop();
}

void test_double_commit_rejected() {
  std::printf("producer protocol violations:\n");

  HostIngressRing ingress(kIngressDepth, kDesc.bytes());

  const uint32_t slot = ingress.acquire();
  ingress.commit(slot, 1, kDesc.bytes());

  // The slot index still matches the most recent acquire, so only the
  // outstanding-count check catches this.
  bool threw = false;
  try {
    ingress.commit(slot, 2, kDesc.bytes());
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "committing the same slot twice is rejected");
  check(ingress.committed() == 1, "the rejected commit did not reach the consumer");

  threw = false;
  try {
    ingress.commit(slot, 3, kDesc.bytes() + 1);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "a frame larger than the slot is rejected");

  ingress.stop();
}

void test_reuse_wait_policies() {
  std::printf("reuse-wait policies:\n");

  // Both policies must be observationally identical from the consumer's side --
  // the difference is only which side of the boundary does the waiting.
  for (auto policy : {perception::ReuseWait::HostSync, perception::ReuseWait::DeviceWait}) {
    const std::string label =
        policy == perception::ReuseWait::HostSync ? "HostSync" : "DeviceWait";

    HostIngressRing ingress(kIngressDepth, kDesc.bytes());
    DeviceRingBuffer device(kDeviceDepth, kDesc.bytes(), policy);
    PassThroughTransform transform;
    UploadStage stage(ingress, device, transform, kDesc);

    check(device.reuse_wait() == policy, label + ": policy is what was asked for");

    bool all_matched = true;
    for (uint8_t n = 1; n <= 20; ++n) {
      if (!stage_frame(ingress, n, 1000ull * n) || !stage.step() ||
          !latest_matches(device, n, 1000ull * n)) {
        all_matched = false;
        break;
      }
    }
    check(all_matched, label + ": 20 frames arrive intact");
    check(stage.failed() == 0, label + ": no failures");
    ingress.stop();
  }

  // The upload-only path acquires the slot before the copy, so it exercises the
  // interlock in the opposite order from the transform path.
  {
    HostIngressRing ingress(kIngressDepth, kDesc.bytes());
    DeviceRingBuffer device(kDeviceDepth, kDesc.bytes(), perception::ReuseWait::DeviceWait);
    UploadStage stage(ingress, device, kDesc);

    bool all_matched = true;
    for (uint8_t n = 1; n <= 20; ++n) {
      if (!stage_frame(ingress, n, 1000ull * n) || !stage.step() ||
          !latest_matches(device, n, 1000ull * n)) {
        all_matched = false;
        break;
      }
    }
    check(all_matched, "DeviceWait: upload-only path arrives intact");
    ingress.stop();
  }
}

// Publish `count` frames into `ring`, each filled with pattern byte n and
// stamped 1000*n, so a leased read can be checked byte for byte.
void publish_frames(DeviceRingBuffer& ring, cudaStream_t stream, uint8_t first, uint32_t count) {
  std::vector<uint8_t> pattern(kDesc.bytes());
  for (uint32_t i = 0; i < count; ++i) {
    const uint8_t n = static_cast<uint8_t>(first + i);
    fill(pattern.data(), pattern.size(), n);
    perception::WriteLease lease = ring.acquire_write(stream);
    cudaMemcpy(lease.data(), pattern.data(), pattern.size(), cudaMemcpyHostToDevice);
    lease.publish(1000ull * n);
  }
}

void test_write_lease() {
  std::printf("write lease:\n");

  DeviceRingBuffer device(kDeviceDepth, kDesc.bytes());
  perception::CudaStream stream;

  // The legacy default stream serialises against unrelated work, so it is
  // refused where the slot is claimed rather than somewhere downstream.
  bool threw = false;
  try {
    device.acquire_write(nullptr);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "the legacy default stream is refused");

  perception::FrameView view;
  {
    perception::WriteLease lease = device.acquire_write(stream);
    check(lease.valid() && lease.data() != nullptr, "a lease hands over a writable slot");
    check(!device.view_latest_inplace(view), "a held slot is not visible to consumers");
    lease.publish(4242);
  }
  check(device.view_latest_inplace(view) && view.frame.timestamp_ns == 4242,
        "publish makes it visible");

  // The abandon path: dropped without publish. Parity has to be restored, or the
  // next lap's acquire would flip an unwritten slot to even.
  const uint32_t abandoned = [&] {
    perception::WriteLease lease = device.acquire_write(stream);
    return lease.slot();
  }();
  check(device.view_latest_inplace(view) && view.frame.timestamp_ns == 4242,
        "an abandoned slot leaves latest_ alone");
  check(!device.get_view_by_timestamp(DeviceRingBuffer::kNoTimestamp, 0, view, false),
        "an abandoned slot carries a timestamp no query can match");

  // Lap the ring so the abandoned slot is reused normally rather than wedged.
  publish_frames(device, stream, 1, kDeviceDepth + 2);
  check(device.view_latest_inplace(view) && view.slot_seq % 2 == 0,
        "the ring keeps publishing after an abandon");
  (void)abandoned;
}

void test_read_lease_blocks_reuse() {
  std::printf("read lease pins its slot against the producer:\n");

  // Depth 2 so a single held lease forces the producer's hand immediately.
  DeviceRingBuffer device(2, kDesc.bytes(), perception::ReuseWait::HostSync,
                          perception::WritePolicy::ScanForFree);
  perception::CudaStream producer;
  perception::CudaStream consumer;

  publish_frames(device, producer, 1, 1);

  perception::ReadLease lease = device.lease_latest(0, consumer);
  check(lease.valid(), "a published frame can be leased");
  check(lease.timestamp_ns() == 1000, "the lease carries the capture timestamp");

  const uint32_t leased_slot = lease.slot();

  // Depth 2, one slot leased, one slot is latest_ -- and latest_ IS the leased
  // one here, so the producer must take the other. It must never take either the
  // leased slot or the current latest.
  {
    perception::WriteLease w = device.acquire_write(producer);
    check(w.slot() != leased_slot, "the producer skips the leased slot");
    w.publish(2000);
  }

  // Contents must be untouched: this is the read the old design could tear.
  std::vector<uint8_t> got(kDesc.bytes());
  cudaMemcpy(got.data(), lease.data(), got.size(), cudaMemcpyDeviceToHost);
  std::vector<uint8_t> want(kDesc.bytes());
  fill(want.data(), want.size(), 1);
  check(std::memcmp(got.data(), want.data(), got.size()) == 0,
        "the leased slot's contents survived a producer lap");

  lease.release();
  check(device.write_stalls() == 0, "the producer never had to stall at this depth");
}

void test_read_lease_stalls_producer() {
  std::printf("producer stalls rather than corrupting:\n");

  // Depth 2: hold one slot, and the only other slot is latest_. The producer has
  // nowhere to go and must wait.
  DeviceRingBuffer device(2, kDesc.bytes(), perception::ReuseWait::HostSync,
                          perception::WritePolicy::ScanForFree);
  perception::CudaStream producer;
  perception::CudaStream consumer;

  publish_frames(device, producer, 1, 2);  // fills both slots; latest_ is the 2nd

  perception::ReadLease lease = device.lease_latest(0, consumer);
  check(lease.valid(), "leased the newest frame");

  std::atomic<bool> wrote{false};
  std::thread writer([&] {
    perception::WriteLease w = device.acquire_write(producer);
    wrote.store(true);
    w.publish(9000);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  check(!wrote.load(), "the producer blocks while the only free slot is leased");

  lease.release();
  writer.join();
  check(wrote.load(), "releasing the lease lets the producer through");
  check(device.write_stalls() > 0, "the stall was counted");
}

void test_lease_by_timestamp() {
  std::printf("timestamp lease, for stereo pairing:\n");

  DeviceRingBuffer device(kDeviceDepth, kDesc.bytes());
  perception::CudaStream stream;
  publish_frames(device, stream, 1, 4);  // timestamps 1000, 2000, 3000, 4000

  {
    perception::ReadLease lease = device.lease_by_timestamp(2000, 10, 0, stream, true);
    check(lease.valid() && lease.timestamp_ns() == 2000, "exact match");
  }
  {
    perception::ReadLease lease = device.lease_by_timestamp(2400, 700, 0, stream, true);
    check(lease.valid() && lease.timestamp_ns() == 2000, "closest match inside tolerance");
  }
  {
    perception::ReadLease lease = device.lease_by_timestamp(9000, 10, 0, stream, true);
    check(!lease.valid(), "nothing inside tolerance yields no lease");
  }
}

void test_lease_validation() {
  std::printf("lease validation:\n");

  DeviceRingBuffer device(kDeviceDepth, kDesc.bytes(), perception::ReuseWait::HostSync,
                          perception::WritePolicy::RoundRobin, /*max_consumers=*/2);
  perception::CudaStream stream;
  publish_frames(device, stream, 1, 1);

  bool threw = false;
  try {
    device.lease_latest(2, stream);  // ids are 0..max_consumers-1
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "a consumer id past max_consumers is rejected");

  threw = false;
  try {
    DeviceRingBuffer too_shallow(1, kDesc.bytes());
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "depth 1 is rejected -- the producer would have to overwrite latest_");

  // Two consumers can hold the same slot at once.
  perception::ReadLease a = device.lease_latest(0, stream);
  perception::ReadLease b = device.lease_latest(1, stream);
  check(a.valid() && b.valid() && a.slot() == b.slot(),
        "two consumers share one slot through a refcount");
}

void test_graph_capture() {
  std::printf("captured graph replay:\n");

  HostIngressRing ingress(kIngressDepth, kDesc.bytes());
  DeviceRingBuffer device(kDeviceDepth, kDesc.bytes());
  PassThroughTransform transform;

  UploadStage::Config config;
  config.use_graph = true;
  UploadStage stage(ingress, device, transform, kDesc, config);

  check(stage.graph_enabled(), "graphs were captured");

  // Same pattern and count as the eager paths, so identical bytes must come out
  // of a replayed graph as out of a per-frame enqueue.
  bool all_matched = true;
  for (uint8_t n = 1; n <= 20; ++n) {
    if (!stage_frame(ingress, n, 1000ull * n) || !stage.step() ||
        !latest_matches(device, n, 1000ull * n)) {
      all_matched = false;
      std::printf("      mismatch at frame %u\n", n);
      break;
    }
  }
  check(all_matched, "20 replayed frames arrive intact, laps every captured graph");
  check(stage.uploaded() == 20, "uploaded counter tracks");
  check(stage.failed() == 0, "no failures");
  check(stage.graph_fallbacks() == 0, "no frame fell back to the eager path");

  ingress.stop();
}

void test_graph_rejects_partial_frame() {
  std::printf("graph replay guards its baked assumptions:\n");

  HostIngressRing ingress(kIngressDepth, kDesc.bytes());
  DeviceRingBuffer device(kDeviceDepth, kDesc.bytes());
  PassThroughTransform transform;

  UploadStage::Config config;
  config.use_graph = true;
  UploadStage stage(ingress, device, transform, kDesc, config);

  // A graph bakes its copy size, so a short frame has to take the eager path
  // rather than replay a copy of the wrong length.
  const uint32_t slot = ingress.acquire();
  fill(ingress.host_ptr(slot), kDesc.bytes(), 7);
  ingress.commit(slot, 7000, kDesc.bytes() / 2);
  check(stage.step(), "a short frame still uploads");
  check(stage.graph_fallbacks() == 1, "it fell back to the eager path");

  ingress.stop();
}

void test_depth_divisibility() {
  std::printf("ring depth divisibility (graph mode only):\n");

  PassThroughTransform transform;
  DeviceRingBuffer device(kDeviceDepth, kDesc.bytes());  // depth 8

  UploadStage::Config graphed;
  graphed.use_graph = true;

  {
    HostIngressRing ingress(3, kDesc.bytes());  // 8 % 3 != 0
    bool threw = false;
    try {
      UploadStage stage(ingress, device, transform, kDesc, graphed);
    } catch (const std::exception&) {
      threw = true;
    }
    check(threw, "graph mode rejects an ingress depth that does not divide the output depth");

    // The eager path recomputes its pointers per frame, so the same depths are
    // fine there -- and must keep working, since that is the default.
    UploadStage eager(ingress, device, transform, kDesc);
    check(!eager.graph_enabled(), "the same depths are accepted without graphs");

    bool all_matched = true;
    for (uint8_t n = 1; n <= 12; ++n) {
      if (!stage_frame(ingress, n, 1000ull * n) || !eager.step() ||
          !latest_matches(device, n, 1000ull * n)) {
        all_matched = false;
        break;
      }
    }
    check(all_matched, "and indivisible depths still deliver frames intact");
    ingress.stop();
  }

  {
    HostIngressRing ingress(kIngressDepth, kDesc.bytes());
    UploadStage::Config config = graphed;
    config.scratch_slots = 3;  // 8 % 3 != 0
    bool threw = false;
    try {
      UploadStage stage(ingress, device, transform, kDesc, config);
    } catch (const std::exception&) {
      threw = true;
    }
    check(threw, "graph mode rejects a scratch depth that does not divide the output depth");

    config.use_graph = false;
    UploadStage eager(ingress, device, transform, kDesc, config);
    check(eager.scratch_slots() == 3, "the eager path accepts scratch_slots = 3");

    // The default config has to satisfy the rule graph mode enforces.
    UploadStage ok(ingress, device, transform, kDesc, graphed);
    check(ok.scratch_slots() == 4 && ok.graph_enabled(),
          "the default scratch depth of 4 divides 8, so graph mode accepts it");
    ingress.stop();
  }

  {
    HostIngressRing ingress(kIngressDepth, kDesc.bytes());
    bool threw = false;
    try {
      UploadStage stage(ingress, device, kDesc, graphed);  // no transform
    } catch (const std::exception&) {
      threw = true;
    }
    check(threw, "graph capture without a transform is rejected");
    ingress.stop();
  }
}

void test_transform_stage_chain() {
  std::printf("ring -> transform -> ring, chained behind an upload:\n");

  HostIngressRing ingress(kIngressDepth, kDesc.bytes());
  DeviceRingBuffer bayer(kDeviceDepth, kDesc.bytes());
  DeviceRingBuffer rgb(kDeviceDepth, kDesc.bytes());
  PassThroughTransform upload_transform;
  PassThroughTransform hop_transform;

  UploadStage upload(ingress, bayer, upload_transform, kDesc);
  TransformStage hop(bayer, rgb, hop_transform, kDesc);

  check(hop.output_desc().bytes() == kDesc.bytes(), "the transform sizes the output ring");

  bool all_matched = true;
  for (uint8_t n = 1; n <= 20; ++n) {
    if (!stage_frame(ingress, n, 1000ull * n) || !upload.step() || !hop.step()) {
      all_matched = false;
      break;
    }
    // Intact at the far end of two hops, with the capture timestamp carried
    // through rather than regenerated.
    if (!latest_matches(rgb, n, 1000ull * n)) {
      all_matched = false;
      std::printf("      mismatch at frame %u\n", n);
      break;
    }
  }
  check(all_matched, "20 frames survive both hops with their timestamps");
  check(hop.transformed() == 20, "transformed counter tracks");
  check(hop.failed() == 0, "no failures");
  check(hop.missed() == 0, "no frame was missed");

  // Latest-wins input: with nothing newly published, there is nothing to do.
  check(!hop.step(), "step() is false when the input ring has published nothing new");

  ingress.stop();
}

void test_transform_stage_threaded() {
  std::printf("transform stage on its own worker:\n");

  constexpr uint32_t kFrames = 200;

  HostIngressRing ingress(kIngressDepth, kDesc.bytes());
  DeviceRingBuffer bayer(kDeviceDepth, kDesc.bytes());
  DeviceRingBuffer rgb(kDeviceDepth, kDesc.bytes());
  PassThroughTransform upload_transform;
  PassThroughTransform hop_transform;

  UploadStage upload(ingress, bayer, upload_transform, kDesc);
  TransformStage hop(bayer, rgb, hop_transform, kDesc);

  upload.start();
  hop.start();

  std::thread source([&] {
    for (uint32_t n = 0; n < kFrames; ++n) {
      if (!stage_frame(ingress, static_cast<uint8_t>(n), 1000ull * (n + 1))) return;
      // Loosely paced so the latest-wins hop is not simply dropping everything;
      // it does not need to see every frame, only the last one.
      std::this_thread::sleep_for(std::chrono::microseconds(300));
    }
  });
  source.join();

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (upload.uploaded() < kFrames && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  // Let the hop catch up to whatever the upload published last.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  hop.stop();
  upload.stop();
  ingress.stop();

  check(upload.uploaded() == kFrames, "every frame was uploaded");
  check(hop.failed() == 0, "no failures in the hop");
  // Latest-wins: the hop is entitled to skip frames, but it must produce some
  // and must land on the newest one.
  check(hop.transformed() > 0 && hop.transformed() <= kFrames,
        "the hop transformed a subset, as latest-wins allows");
  check(latest_matches(rgb, static_cast<uint8_t>(kFrames - 1), 1000ull * kFrames),
        "the newest frame made it through both hops");
  std::printf("      transformed %lu of %u, overruns %lu\n",
              static_cast<unsigned long>(hop.transformed()), kFrames,
              static_cast<unsigned long>(hop.missed()));
}

void test_transform_stage_validation() {
  std::printf("transform stage startup validation:\n");

  DeviceRingBuffer a(kDeviceDepth, kDesc.bytes());
  PassThroughTransform transform;

  bool threw = false;
  try {
    TransformStage self(a, a, transform, kDesc);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "a stage from a ring to itself is rejected");

  DeviceRingBuffer too_small(kDeviceDepth, kDesc.bytes() - 1);
  threw = false;
  try {
    TransformStage stage(a, too_small, transform, kDesc);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "an output ring too small for the transform is rejected");
}

void test_size_validation() {
  std::printf("startup validation:\n");

  HostIngressRing ingress(kIngressDepth, kDesc.bytes());
  PassThroughTransform transform;

  // One byte short of what PassThrough produces.
  DeviceRingBuffer too_small(kDeviceDepth, kDesc.bytes() - 1);
  bool threw = false;
  try {
    UploadStage stage(ingress, too_small, transform, kDesc);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "an output ring too small for the transform is rejected at construction");

  threw = false;
  try {
    UploadStage stage(ingress, too_small, kDesc);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "the upload-only path validates the output ring too");

  ingress.stop();
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("no CUDA device available\n");
    return 1;
  }

  try {
    test_inline_step();
    test_no_transform();
    test_no_transform_worker();
    test_worker_thread();
    test_backpressure();
    test_stop_unblocks_producer();
    test_abandon();
    test_double_commit_rejected();
    test_reuse_wait_policies();
    test_write_lease();
    test_read_lease_blocks_reuse();
    test_read_lease_stalls_producer();
    test_lease_by_timestamp();
    test_lease_validation();
    test_graph_capture();
    test_graph_rejects_partial_frame();
    test_depth_divisibility();
    test_transform_stage_chain();
    test_transform_stage_threaded();
    test_transform_stage_validation();
    test_size_validation();
  } catch (const std::exception& e) {
    std::printf("threw: %s\n", e.what());
    return 1;
  }

  std::printf("\n%s\n", g_failures == 0 ? "all checks passed" : "FAILURES");
  return g_failures == 0 ? 0 : 1;
}
