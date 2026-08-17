#include "viewer_consumer.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <stdexcept>

#include "cuda_util.hpp"
#include "gl_viewer.hpp"

namespace perception {
namespace {

// How long the thread sits in the window's event wait before looking at the
// ring again. Only bounds how stale the debug window can get; a frame does not
// wake it, since it is parked on the display connection rather than the ring.
constexpr double kPollSeconds = 0.005;

// Frames between the viewer's own throughput lines. It reports for itself
// rather than through the pipeline's stats line, which would mean another
// thread reading its counters.
constexpr uint64_t kReportEvery = 60;

}  // namespace

ViewerConsumer::ViewerConsumer(DeviceRingBuffer& ring, const LatencyProbe& probe,
                               const ImageDesc& desc, const DisplayConfig& config,
                               uint32_t consumer_id, int device_id)
    : ring_(&ring),
      probe_(&probe),
      desc_(desc),
      config_(config),
      consumer_id_(consumer_id),
      device_id_(device_id) {
  if (consumer_id_ >= ring_->max_consumers()) {
    throw std::runtime_error(
        "ViewerConsumer: consumer id is outside the ring's max_consumers; raise "
        "pipeline.max_consumers or disable the display");
  }
}

ViewerConsumer::~ViewerConsumer() { stop(); }

void ViewerConsumer::start() {
  if (!config_.enable) return;
  if (running_.exchange(true)) return;
  thread_ = std::thread(&ViewerConsumer::run, this);
}

void ViewerConsumer::stop() {
  running_.store(false, std::memory_order_relaxed);
  if (thread_.joinable()) thread_.join();
}

void ViewerConsumer::run() {
  GlViewer::Config view_config;
  view_config.window_width = config_.window_width;
  view_config.window_height = config_.window_height;
  view_config.vsync = config_.vsync;
  view_config.latency_scale_ms = config_.latency_scale_ms;

  // Constructed here and nowhere else: the GL context is thread-current and the
  // CUDA-GL interop calls need it bound, so the window is created, pumped,
  // drawn with, and destroyed all on this thread.
  std::unique_ptr<GlViewer> viewer;
  try {
    viewer = std::make_unique<GlViewer>(desc_, view_config);
    std::printf("display: %ux%u vsync=%s scale=%.0fms (esc/q to quit)\n", config_.window_width,
                config_.window_height, config_.vsync ? "on" : "off", config_.latency_scale_ms);
  } catch (const std::exception& e) {
    // Deliberately not a close: a headless box should keep running the
    // pipeline, and closed() staying false is what lets it.
    std::printf("display: disabled (%s)\n", e.what());
    return;
  }

  // Per-thread CUDA state, like every other worker sets for itself.
  if (cudaSetDevice(device_id_) != cudaSuccess) {
    std::printf("display: disabled (cudaSetDevice failed on the viewer thread)\n");
    return;
  }

  CudaStream stream;
  uint32_t last_slot = ~0u;
  uint64_t last_seq = ~0ull;

  try {
    while (running_.load(std::memory_order_relaxed)) {
      // Blocks on the display connection, so an idle window costs nothing. The
      // ring is checked afterwards on the same tick.
      viewer->poll_wait(kPollSeconds);
      if (viewer->should_close()) break;

      // Unleased peek first. A lease taken only to discover nothing is new
      // still bumps the slot's reader count, and that stalls the producer in
      // wait_unleased for a read that never happens.
      FramePeek peek;
      if (!ring_->view_latest_inplace(peek) ||
          (peek.slot == last_slot && peek.slot_seq == last_seq)) {
        continue;
      }

      ReadLease lease = ring_->lease_latest(consumer_id_, stream);
      if (!lease.valid()) continue;
      last_slot = lease.slot();
      last_seq = lease.seq();

      // Queue-side ordering only -- returns immediately, no CPU block.
      cuda_error_check(cudaStreamWaitEvent(stream, lease.data_ready_event(), 0),
                       "ViewerConsumer: cudaStreamWaitEvent");

      double latency_ms = -1.0;
      int64_t age_ns = 0;
      if (probe_->age_ns(lease.timestamp_ns(), age_ns)) {
        latency_ms = static_cast<double>(age_ns) * 1e-6;
      }

      // The lease still holds the slot, so the copy reads stable memory; the
      // lease's destructor records read completion on this same stream.
      viewer->present(lease.data(), stream, latency_ms);

      if (viewer->presented() % kReportEvery == 0) {
        std::printf("display: presented=%lu present %.2f ms\n", viewer->presented(),
                    viewer->last_present_ms());
      }
    }
  } catch (const std::exception& e) {
    std::printf("display: stopped (%s)\n", e.what());
  }

  // The window really did exist and is going away, so whoever is parked on the
  // ring has to be told -- both that it happened and that there is something to
  // re-check.
  closed_.store(true, std::memory_order_relaxed);
  ring_->wake_all();

  // Nothing may still be reading a slot when the lease bookkeeping goes.
  cudaStreamSynchronize(stream);
}

}  // namespace perception
