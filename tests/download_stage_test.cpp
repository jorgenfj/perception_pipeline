// Behaviour test for the device-to-host boundary. Needs a GPU but no camera and
// no engine: the "producer" is this file, writing a known pattern into device
// memory and asking the stage to bring it back.
//
// What is worth testing here is not that cudaMemcpyAsync works. It is the three
// promises the stage makes to a caller that cannot afford to block: that a
// frame is published only once its copy has actually retired, that the pool
// bounds the damage when a sink is slow instead of corrupting a slot, and that
// a frame outliving the pool is safe rather than a use-after-free.

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cuda_util.hpp"
#include "download_stage.hpp"
#include "host_frame_pool.hpp"
#include "latest_frame.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 32;
constexpr std::size_t kFrameBytes = static_cast<std::size_t>(kWidth) * kHeight * sizeof(float);

// A pattern that is different for every frame and cheap to verify.
std::vector<float> make_pattern(uint32_t frame) {
  std::vector<float> out(static_cast<std::size_t>(kWidth) * kHeight);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<float>(frame * 1000 + (i % 97));
  }
  return out;
}

bool matches(const perception::HostFrame& got, const std::vector<float>& want) {
  return std::memcmp(got.data, want.data(), want.size() * sizeof(float)) == 0;
}

void test_pool_bounds_and_recycles() {
  std::printf("pool: bounded, and slots come back\n");
  perception::HostFramePool pool(3, kFrameBytes, kWidth, kHeight);

  check(pool.slots() == 3 && pool.pinned_bytes() == 3 * kFrameBytes, "allocates what it was asked");

  auto a = pool.acquire(100);
  auto b = pool.acquire(200);
  auto c = pool.acquire(300);
  check(a && b && c, "hands out every slot");
  check(pool.in_use() == 3, "and knows they are out");

  check(pool.acquire(400) == nullptr, "returns null rather than overcommitting");
  check(pool.drops() == 1, "and counts that as a drop");

  // Distinct storage: two frames aliasing the same buffer is the failure this
  // whole design exists to prevent.
  check(a->data != b->data && b->data != c->data && a->data != c->data,
        "every slot is its own buffer");
  check(a->timestamp_ns == 100 && b->timestamp_ns == 200, "carries the stamp it was given");
  check(a->sequence != b->sequence, "sequence is unique per frame");

  b.reset();
  check(pool.in_use() == 2, "dropping a frame returns its slot");
  auto d = pool.acquire(500);
  check(d != nullptr, "which the next acquire can have");
  check(pool.peak_in_use() == 3, "peak survives the dip");
}

void test_frame_outlives_pool() {
  std::printf("pool: a frame may outlive the pool that issued it\n");
  std::shared_ptr<const perception::HostFrame> escaped;
  {
    perception::HostFramePool pool(2, kFrameBytes, kWidth, kHeight);
    auto frame = pool.acquire(42);
    std::memcpy(frame->data, make_pattern(7).data(), kFrameBytes);
    escaped = frame;
  }
  // The pool is gone; a sink still draining its queue during shutdown is
  // exactly this case, and it must not be a use-after-free.
  check(escaped != nullptr && matches(*escaped, make_pattern(7)),
        "the bytes are still readable after the pool is destroyed");
  escaped.reset();
  check(true, "and releasing it afterwards does not fault");
}

void test_round_trip_and_ordering() {
  std::printf("stage: frames come back whole, in order, only once retired\n");

  perception::DownloadStage::Config config;
  config.slots = 4;
  config.frame_bytes = kFrameBytes;
  config.width = kWidth;
  config.height = kHeight;
  perception::DownloadStage stage(config);

  std::vector<std::shared_ptr<const perception::HostFrame>> got;
  std::mutex got_mutex;
  stage.add_sink([&](const std::shared_ptr<const perception::HostFrame>& frame) {
    const std::lock_guard<std::mutex> lock(got_mutex);
    got.push_back(frame);
  });
  stage.start();

  void* device = nullptr;
  perception::cuda_error_check(cudaMalloc(&device, kFrameBytes), "cudaMalloc");
  perception::CudaStream stream;

  constexpr uint32_t kFrames = 3;
  for (uint32_t i = 0; i < kFrames; ++i) {
    const std::vector<float> pattern = make_pattern(i);
    perception::cuda_error_check(
        cudaMemcpyAsync(device, pattern.data(), kFrameBytes, cudaMemcpyHostToDevice, stream),
        "H2D");
    check(stage.enqueue(device, stream, 1000 + i), "enqueue accepted frame " + std::to_string(i));
    // The next H2D would overwrite `device`, so the stage's copy has to be
    // ordered ahead of it. Same-stream ordering is what guarantees that, and
    // this loop is the thing that would break if it did not.
    perception::cuda_error_check(cudaStreamSynchronize(stream), "sync");
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  for (;;) {
    {
      const std::lock_guard<std::mutex> lock(got_mutex);
      if (got.size() >= kFrames) break;
    }
    if (std::chrono::steady_clock::now() > deadline) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }

  const std::lock_guard<std::mutex> lock(got_mutex);
  check(got.size() == kFrames, "every enqueued frame was published");
  if (got.size() == kFrames) {
    bool bytes_ok = true;
    bool order_ok = true;
    bool stamps_ok = true;
    for (uint32_t i = 0; i < kFrames; ++i) {
      if (!matches(*got[i], make_pattern(i))) bytes_ok = false;
      if (got[i]->timestamp_ns != 1000 + i) stamps_ok = false;
      if (i > 0 && got[i]->sequence <= got[i - 1]->sequence) order_ok = false;
    }
    check(bytes_ok, "every payload came back byte for byte");
    check(order_ok, "published in the order they were enqueued");
    check(stamps_ok, "each frame kept the timestamp it was enqueued with");
    check(got[0]->host_ready_ns > 0, "host_ready_ns is stamped at publish");
  }
  check(stage.drops() == 0, "nothing dropped with a pool deeper than the burst");
  check(stage.enqueued() == kFrames && stage.published() == kFrames, "counters agree");

  stage.stop();
  cudaFree(device);
}

void test_latest_frame_holder_takes_the_newest() {
  std::printf("stage: a LatestFrame sink hands the consumer the newest, and pins one slot\n");

  perception::DownloadStage::Config config;
  config.slots = 4;
  config.frame_bytes = kFrameBytes;
  config.width = kWidth;
  config.height = kHeight;
  perception::DownloadStage stage(config);

  // The composition being tested: the holder's sink() is a DownloadStage::Sink,
  // and the consumer pulls rather than being called.
  perception::LatestFrame latest("consumer");
  stage.add_sink(latest.sink());
  stage.start();

  void* device = nullptr;
  perception::cuda_error_check(cudaMalloc(&device, kFrameBytes), "cudaMalloc");
  perception::CudaStream stream;

  constexpr uint32_t kFrames = 3;
  for (uint32_t i = 0; i < kFrames; ++i) {
    const std::vector<float> pattern = make_pattern(i);
    perception::cuda_error_check(
        cudaMemcpyAsync(device, pattern.data(), kFrameBytes, cudaMemcpyHostToDevice, stream),
        "H2D");
    check(stage.enqueue(device, stream, 2000 + i), "enqueue accepted frame " + std::to_string(i));
    perception::cuda_error_check(cudaStreamSynchronize(stream), "sync");
  }

  // Nobody has looked yet, so the last one to arrive is the one waiting and the
  // two before it are skips -- the point of the holder, not a fault.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (latest.offered() < kFrames && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }

  const auto frame = latest.acquire_latest();
  check(frame != nullptr, "the consumer got a frame");
  if (frame) {
    check(matches(*frame, make_pattern(kFrames - 1)), "and it is the newest, byte for byte");
    check(frame->timestamp_ns == 2000 + kFrames - 1, "with the stamp it was enqueued with");
  }
  check(latest.skipped() == kFrames - 1, "the ones it stepped over are counted");

  // One slot in the consumer's hand, and nothing else retained: the two skipped
  // frames went back to the pool as they were displaced.
  check(stage.pool().in_use() == 1, "the holder pins one slot, not one per frame");
  check(stage.drops() == 0, "and a pool this deep never ran dry");

  stage.stop();
  cudaFree(device);
}

void test_slow_sink_drops_rather_than_corrupts() {
  std::printf("stage: a sink holding every slot makes enqueue drop, not overwrite\n");

  perception::DownloadStage::Config config;
  config.slots = 2;
  config.frame_bytes = kFrameBytes;
  config.width = kWidth;
  config.height = kHeight;
  perception::DownloadStage stage(config);

  // The pathological sink: it never lets go. This is a stand-in for a disk that
  // has stopped keeping up, and the contract is that the producer keeps running
  // and the frames are counted as lost -- never that a slot is reused under a
  // reader.
  std::vector<std::shared_ptr<const perception::HostFrame>> hoarded;
  std::mutex hoard_mutex;
  stage.add_sink([&](const std::shared_ptr<const perception::HostFrame>& frame) {
    const std::lock_guard<std::mutex> lock(hoard_mutex);
    hoarded.push_back(frame);
  });
  stage.start();

  void* device = nullptr;
  perception::cuda_error_check(cudaMalloc(&device, kFrameBytes), "cudaMalloc");
  perception::CudaStream stream;

  const std::vector<float> pattern = make_pattern(11);
  perception::cuda_error_check(
      cudaMemcpyAsync(device, pattern.data(), kFrameBytes, cudaMemcpyHostToDevice, stream), "H2D");
  perception::cuda_error_check(cudaStreamSynchronize(stream), "sync");

  uint32_t accepted = 0;
  uint32_t refused = 0;
  for (uint32_t i = 0; i < 12; ++i) {
    if (stage.enqueue(device, stream, 2000 + i)) {
      ++accepted;
    } else {
      ++refused;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  perception::cuda_error_check(cudaStreamSynchronize(stream), "sync");

  check(refused > 0, "enqueue refuses once the pool is exhausted");
  check(stage.drops() == refused, "every refusal is counted as a drop");
  check(accepted <= config.slots + 1, "no more than the pool depth was ever in flight");

  {
    const std::lock_guard<std::mutex> lock(hoard_mutex);
    bool intact = true;
    for (const auto& frame : hoarded) {
      if (!matches(*frame, pattern)) intact = false;
    }
    check(intact, "frames the sink is still holding were never overwritten");
  }

  stage.stop();
  cudaFree(device);
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("no CUDA device -- skipping\n");
    return 0;
  }

  try {
    test_pool_bounds_and_recycles();
    test_frame_outlives_pool();
    test_round_trip_and_ordering();
    test_latest_frame_holder_takes_the_newest();
    test_slow_sink_drops_rather_than_corrupts();
  } catch (const std::exception& e) {
    std::printf("  [FAIL] threw: %s\n", e.what());
    ++g_failures;
  }

  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
