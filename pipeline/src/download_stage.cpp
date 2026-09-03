#include "download_stage.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace perception {
namespace {

// Deliberately spelled out here rather than including capture's frame_sink.hpp:
// perception_core knows nothing about cameras or config, and perception_capture
// would drag geometry and Eigen in behind it for one clock read. Must stay
// system_clock to match host_now_ns() and LatencyProbe -- a steady_clock stamp
// shares no epoch with a camera's, and host_ready_ns - timestamp_ns would stop
// meaning latency.
uint64_t host_realtime_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count());
}

// Long enough that an idle stage costs nothing, short enough that the extra
// latency it adds to a retired copy is far under a frame period at any rate
// this pipeline runs.
constexpr std::chrono::microseconds kPollInterval{500};

}  // namespace

DownloadStage::DownloadStage(const Config& config)
    : config_(config),
      pool_(config.slots, config.frame_bytes, config.width, config.height, config.device_id) {
  events_.reserve(config.slots);
  for (uint32_t i = 0; i < config.slots; ++i) {
    events_.push_back(std::make_unique<CudaEvent>(cudaEventDisableTiming));
    free_events_.push_back(i);
  }
}

DownloadStage::~DownloadStage() { stop(); }

void DownloadStage::add_sink(Sink sink) {
  if (running_.load(std::memory_order_relaxed)) {
    throw std::runtime_error("DownloadStage: add_sink() after start()");
  }
  sinks_.push_back(std::move(sink));
}

void DownloadStage::start() {
  if (running_.exchange(true, std::memory_order_relaxed)) return;
  thread_ = std::thread([this] { run(); });
}

void DownloadStage::stop() {
  if (!running_.exchange(false, std::memory_order_relaxed)) return;
  queued_.notify_all();
  if (thread_.joinable()) thread_.join();

  // Whatever is still in flight is dropped rather than waited on: stop() runs
  // during teardown, the producer's stream may already be gone, and a frame
  // that has not retired has nothing to publish. The shared_ptrs go with the
  // deque and the slots return to the pool.
  const std::lock_guard<std::mutex> lock(mutex_);
  inflight_.clear();
}

bool DownloadStage::enqueue(const void* device_src, cudaStream_t stream, uint64_t timestamp_ns) {
  reject_default_stream(stream, "DownloadStage::enqueue");

  // Reserve first. A copy enqueued with nowhere to land cannot be taken back.
  std::shared_ptr<HostFrame> frame = pool_.acquire(timestamp_ns);
  if (!frame) return false;

  uint32_t event = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (free_events_.empty()) {
      // Unreachable: events_ is as deep as the pool, so a free slot implies a
      // free event. Belt and braces -- dropping the frame here is still correct.
      return false;
    }
    event = free_events_.front();
    free_events_.pop_front();
  }

  // On the caller's stream, so it is ordered behind whatever produced the data
  // and ahead of whatever overwrites it next.
  cuda_error_check(cudaMemcpyAsync(frame->data, device_src, frame->bytes, cudaMemcpyDeviceToHost,
                                   stream),
                   "DownloadStage: cudaMemcpyAsync");
  cuda_error_check(cudaEventRecord(*events_[event], stream), "DownloadStage: cudaEventRecord");

  {
    const std::lock_guard<std::mutex> lock(mutex_);
    inflight_.push_back(InFlight{std::move(frame), event});
  }
  queued_.notify_one();
  enqueued_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void DownloadStage::run() {
  if (cudaSetDevice(config_.device_id) != cudaSuccess) {
    std::printf("download: disabled (cudaSetDevice failed on the stage thread)\n");
    return;
  }

  while (running_.load(std::memory_order_relaxed)) {
    std::shared_ptr<HostFrame> ready;
    uint32_t done_event = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (inflight_.empty()) {
        queued_.wait_for(lock, kPollInterval);
        continue;
      }

      // Front only: the copies were enqueued on one stream, so they retire in
      // order and anything behind an unretired copy is unretired too. Checking
      // the rest would be work that cannot pay off.
      InFlight& front = inflight_.front();
      if (cudaEventQuery(*events_[front.event]) != cudaSuccess) {
        cudaGetLastError();  // swallow the cudaErrorNotReady we just provoked
        lock.unlock();
        std::this_thread::sleep_for(kPollInterval);
        continue;
      }

      ready = std::move(front.frame);
      done_event = front.event;
      inflight_.pop_front();
      free_events_.push_back(done_event);
    }

    ready->host_ready_ns = host_realtime_ns();
    published_.fetch_add(1, std::memory_order_relaxed);

    // Outside the lock: a sink may do anything, including block on a disk, and
    // it must not do so while holding up enqueue(). Const from here on -- every
    // sink sees the same frame and none of them owns it.
    const std::shared_ptr<const HostFrame> frame = std::move(ready);
    for (const Sink& sink : sinks_) sink(frame);
  }
}

std::string DownloadStage::health_line() const {
  char line[192];
  std::snprintf(line, sizeof(line), "download: %llu in, %llu out, drops=%llu, pool %u/%u (peak %u)",
                static_cast<unsigned long long>(enqueued()),
                static_cast<unsigned long long>(published()),
                static_cast<unsigned long long>(drops()), pool_.in_use(), pool_.slots(),
                pool_.peak_in_use());
  return line;
}

}  // namespace perception
