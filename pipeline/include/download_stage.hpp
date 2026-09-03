#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cuda_util.hpp"
#include "host_frame_pool.hpp"

namespace perception {

// The device-to-host boundary: device memory -> D2H -> pinned pool -> sinks.
//
// The mirror of UploadStage, which runs pinned ingress -> H2D -> device ring.
// This is where a lease stops being the right abstraction: upstream of here a
// consumer must track GPU completion to know when a slot is reusable,
// downstream of here it holds a shared_ptr and is done. Sinks registered on
// this stage need no consumer_id, no CUDA event, and no place in a ring's
// max_consumers budget, which is what makes them cheap to add.
//
// Nothing here blocks the producer. enqueue() reserves a slot, enqueues an
// async copy and records an event; a private thread polls those events and
// publishes only frames whose copy has actually retired. No cudaStreamSynchronize
// anywhere, so this is safe to call from a pair callback holding ring leases,
// and from a GL thread.
class DownloadStage {
 public:
  // Called on the stage's own thread, once per frame, in the order the copies
  // were enqueued. Sinks may hold the frame for as long as they like -- that is
  // the point -- but a sink that holds more than the pool is deep will make
  // enqueue() start dropping.
  using Sink = std::function<void(const std::shared_ptr<const HostFrame>&)>;

  struct Config {
    uint32_t slots = 4;
    std::size_t frame_bytes = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    int device_id = 0;
  };

  explicit DownloadStage(const Config& config);
  ~DownloadStage();

  DownloadStage(const DownloadStage&) = delete;
  DownloadStage& operator=(const DownloadStage&) = delete;

  // Register before start(). Not thread-safe against start(), deliberately:
  // sinks are wired once at construction time, not swapped at runtime.
  void add_sink(Sink sink);

  void start();
  void stop();

  // Copy `device_src` into a pooled host buffer, on `stream`.
  //
  // Call it on the thread and stream that produced the data, with the copy
  // enqueued behind the work that wrote it -- for a TensorRT output that means
  // immediately after infer() returns, so the copy is ordered ahead of the next
  // cycle overwriting the binding. Same-stream ordering is the whole guarantee;
  // there is no lock on the device memory and none is needed.
  //
  // False means the pool was empty and nothing was enqueued. That is a drop and
  // it is counted. The slot is reserved BEFORE the copy is enqueued, never
  // after: a copy with nowhere to land would either scribble on a slot a sink
  // is reading or have to be abandoned mid-stream, and neither is recoverable.
  bool enqueue(const void* device_src, cudaStream_t stream, uint64_t timestamp_ns);

  uint64_t enqueued() const { return enqueued_.load(std::memory_order_relaxed); }
  uint64_t published() const { return published_.load(std::memory_order_relaxed); }

  // Frames lost because the pool was empty when enqueue() was called.
  uint64_t drops() const { return pool_.drops(); }

  const HostFramePool& pool() const { return pool_; }

  // "download: 780 in, 780 out, drops=0, pool 1/4 (peak 2)"
  std::string health_line() const;

 private:
  void run();

  // One in-flight copy: the frame it lands in, and the event that says so.
  struct InFlight {
    std::shared_ptr<HostFrame> frame;
    uint32_t event = 0;  // index into events_
  };

  Config config_;
  HostFramePool pool_;

  // One per slot -- the pool bounds how many copies can be in flight, so this
  // can never run short. Held by pointer because CudaEvent is neither copyable
  // nor movable (deliberately: it owns a cudaEvent_t), and a vector needs one
  // or the other.
  std::vector<std::unique_ptr<CudaEvent>> events_;

  mutable std::mutex mutex_;
  std::condition_variable queued_;
  std::deque<uint32_t> free_events_;
  std::deque<InFlight> inflight_;  // FIFO: stream order is retirement order

  std::vector<Sink> sinks_;
  std::thread thread_;
  std::atomic<bool> running_{false};

  std::atomic<uint64_t> enqueued_{0};
  std::atomic<uint64_t> published_{0};
};

}  // namespace perception
