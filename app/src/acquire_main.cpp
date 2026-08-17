// Ingress driver: one camera -> pinned ring (as the camera's own buffers) ->
// UploadStage (H2D + debayer) -> device ring, with a local GL viewer leasing
// frames off the ring for latency inspection.
//
//   acquire [config.yaml]
//
// This is the composition root: spinnaker/ and pipeline/ know nothing about
// each other, and everything that joins them lives here.

#include <Spinnaker.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "app_config.hpp"
#include "cuda_util.hpp"
#include "device_ring_buffer.hpp"
#include "host_ingress_ring.hpp"
#include "latency_probe.hpp"
#include "ring_frame_sink.hpp"
#include "spinnaker_source.hpp"
#include "transforms/debayer.hpp"
#include "upload_stage.hpp"

#ifdef PERCEPTION_WITH_DISPLAY
#include "gl_viewer.hpp"
#endif

namespace {

std::atomic<bool> g_stop{false};
sigset_t stop_signals;

constexpr const char* kDefaultConfig = "config/acquire.yaml";
constexpr double kViewerPollSeconds = 0.005;

using perception::cuda_error_check;

// Latency over one reporting interval. Min and max earn their place over a
// plain mean: the mean hides exactly the tail that matters when the question is
// whether a ring is deep enough.
struct LatencyWindow {
  double min_ms = 1e30;
  double max_ms = -1e30;
  double sum_ms = 0.0;
  uint64_t count = 0;

  void add(double ms) {
    min_ms = std::min(min_ms, ms);
    max_ms = std::max(max_ms, ms);
    sum_ms += ms;
    ++count;
  }
  double mean_ms() const { return count ? sum_ms / static_cast<double>(count) : 0.0; }
  void reset() { *this = LatencyWindow{}; }
};

}  // namespace

int main(int argc, char** argv) {
  sigemptyset(&stop_signals);
  sigaddset(&stop_signals, SIGINT);
  if (pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr) != 0) {
    std::printf("FAILED: could not block SIGINT\n");
    return 1;
  }

  const std::string config_path = argc > 1 ? argv[1] : kDefaultConfig;

  Spinnaker::SystemPtr system = Spinnaker::System::GetInstance();
  Spinnaker::CameraList cameras = system->GetCameras();

  int status = 0;
  try {
    const perception::AppConfig config = perception::load_app_config(config_path);
    std::printf("config: %s\n", config_path.c_str());

    perception::SpinnakerSource source(
        perception::SpinnakerSource::select(cameras, config.camera.serial), config.camera);

    const perception::CameraGeometry& geometry = source.geometry();
    const perception::ImageDesc desc = perception::to_image_desc(geometry);
    std::printf("camera: %ux%u stride=%u %s, %zu bytes/frame, %zu bytes/buffer, min %u buffers\n",
                geometry.width, geometry.height, geometry.stride_bytes,
                geometry.pixel_format.c_str(), geometry.frame_bytes, geometry.buffer_bytes,
                source.min_slot_count());

    perception::DebayerTransform debayer;
    const perception::ImageDesc display_desc = debayer.output_desc(desc);
    std::printf("transform: %s -> %ux%u stride=%u RGBA8, %zu bytes/frame\n", debayer.name(),
                display_desc.width, display_desc.height, display_desc.stride_bytes,
                display_desc.bytes());

    perception::LatencyProbe probe;

    perception::HostIngressRing ingress(config.pipeline.ingress_depth, geometry.buffer_bytes,
                                        config.pipeline.device_id, perception::FillMode::External);
    perception::RingFrameSink sink(ingress, &probe);

    perception::DeviceRingBuffer device_ring(
        config.pipeline.device_depth, display_desc.bytes(), config.pipeline.reuse_wait,
        config.pipeline.write_policy, config.pipeline.max_consumers, config.pipeline.device_id);

    // With a transform the H2D lands in scratch and the debayer writes the
    // output slot, so the camera's buffer is released one step earlier than on
    // the upload-only path.
    perception::UploadStage upload(ingress, device_ring, debayer, desc, config.upload);

    std::thread signal_relay([&device_ring] {
      int signal_number = 0;
      sigwait(&stop_signals, &signal_number);
      g_stop.store(true, std::memory_order_relaxed);
      device_ring.wake_all();
    });

    // Set by the viewer thread when the window is closed, so main leaves too.
    std::atomic<bool> viewer_closed{false};

#ifdef PERCEPTION_WITH_DISPLAY
    // The viewer is a second, independent consumer of the ring rather than
    // something main hands frames to: the ring is already multi-consumer, and
    // that keeps the debug window off the pipeline's critical path entirely --
    // it leases on its own id and its own stream, and if it falls behind it
    // simply misses frames.
    //
    // It owns the window end to end. Everything GL has to happen on the thread
    // that created the context, so the GlViewer is constructed, pumped, drawn
    // with, and destroyed here and nowhere else.
    std::thread viewer_thread;
    if (config.display.enable) {
      viewer_thread = std::thread([&] {
        perception::GlViewer::Config view_config;
        view_config.window_width = config.display.window_width;
        view_config.window_height = config.display.window_height;
        view_config.vsync = config.display.vsync;
        view_config.latency_scale_ms = config.display.latency_scale_ms;

        std::unique_ptr<perception::GlViewer> viewer;
        try {
          viewer = std::make_unique<perception::GlViewer>(display_desc, view_config);
          std::printf("display: %ux%u vsync=%s scale=%.0fms (esc/q to quit)\n",
                      config.display.window_width, config.display.window_height,
                      config.display.vsync ? "on" : "off", config.display.latency_scale_ms);
        } catch (const std::exception& e) {
          // Not fatal: the pipeline is the point, the window is an inspector.
          std::printf("display: disabled (%s)\n", e.what());
          return;
        }

        // Per-thread CUDA state, like every other worker sets for itself.
        if (cudaSetDevice(config.pipeline.device_id) != cudaSuccess) {
          std::printf("display: disabled (cudaSetDevice failed on the viewer thread)\n");
          return;
        }
        perception::CudaStream view_stream;
        uint32_t view_slot = ~0u;
        uint64_t view_seq = ~0ull;

        try {
          // viewer_closed is also an inbound signal: main sets it when it is
          // tearing down, which is how this thread learns to leave on the paths
          // where the window itself was never closed.
          while (!g_stop.load(std::memory_order_relaxed) &&
                 !viewer_closed.load(std::memory_order_relaxed)) {
            // Blocks on the display connection, so an idle window costs
            // nothing. The ring is checked afterwards on the same tick.
            viewer->poll_wait(kViewerPollSeconds);
            if (viewer->should_close()) break;

            perception::FramePeek peek;
            if (!device_ring.view_latest_inplace(peek) ||
                (peek.slot == view_slot && peek.slot_seq == view_seq)) {
              continue;
            }

            perception::ReadLease lease = device_ring.lease_latest(1, view_stream);
            if (!lease.valid()) continue;
            view_slot = lease.slot();
            view_seq = lease.seq();

            cuda_error_check(cudaStreamWaitEvent(view_stream, lease.data_ready_event(), 0),
                             "cudaStreamWaitEvent(viewer)");

            double latency_ms = -1.0;
            int64_t age_ns = 0;
            if (probe.age_ns(lease.timestamp_ns(), age_ns)) {
              latency_ms = static_cast<double>(age_ns) * 1e-6;
            }

            viewer->present(lease.data(), view_stream, latency_ms);

            if (viewer->presented() % 60 == 0) {
              std::printf("display: presented=%lu present %.2f ms\n", viewer->presented(),
                          viewer->last_present_ms());
            }
          }
        } catch (const std::exception& e) {
          std::printf("display: stopped (%s)\n", e.what());
        }

        // Whichever way the loop ended, main is waiting on the ring and has to
        // be told the window is gone.
        viewer_closed.store(true, std::memory_order_relaxed);
        device_ring.wake_all();
        cudaStreamSynchronize(view_stream);
      });
    }
#endif

    auto stop_helpers = [&] {
      viewer_closed.store(true, std::memory_order_relaxed);
      device_ring.wake_all();
#ifdef PERCEPTION_WITH_DISPLAY
      if (viewer_thread.joinable()) viewer_thread.join();
#endif
      if (signal_relay.joinable()) {
        pthread_kill(signal_relay.native_handle(), SIGINT);
        signal_relay.join();
      }
    };

    uint64_t consumed = 0;
    uint32_t last_slot = ~0u;
    uint64_t last_seq = ~0ull;
    LatencyWindow latency;

    perception::CudaStream consumer;
    try {
      source.start(sink);
      upload.start();

      while (!g_stop.load(std::memory_order_relaxed) &&
                 !viewer_closed.load(std::memory_order_relaxed)) {
        const uint64_t seen = device_ring.wait_seq();

        perception::FramePeek peek;
        if (!device_ring.view_latest_inplace(peek) ||
            (peek.slot == last_slot && peek.slot_seq == last_seq)) {
          device_ring.wait_for_publish(seen);
          continue;
        }

        perception::ReadLease lease = device_ring.lease_latest(0, consumer);
        if (!lease.valid()) continue;
        last_slot = lease.slot();
        last_seq = lease.seq();

        cuda_error_check(cudaStreamWaitEvent(consumer, lease.data_ready_event(), 0),
                         "cudaStreamWaitEvent");

        // Measured before the draw, so it covers capture -> ready-to-present
        // and excludes the swap. See LatencyProbe for what the number means.
        double latency_ms = -1.0;
        int64_t age_ns = 0;
        if (probe.age_ns(lease.timestamp_ns(), age_ns)) {
          latency_ms = static_cast<double>(age_ns) * 1e-6;
          latency.add(latency_ms);
        }

        (void)latency_ms;

        ++consumed;

        if (consumed % 60 == 0) {
          std::printf("t=%luns slot=%u delivered=%lu uploaded=%lu consumed=%lu "
                      "incomplete=%lu foreign=%lu timeouts=%lu stalls=%lu failed=%lu",
                      lease.timestamp_ns(), lease.slot(), source.delivered(), upload.uploaded(),
                      consumed, source.incomplete(), source.foreign(), source.timeouts(),
                      device_ring.write_stalls(), upload.failed());
          if (latency.count) {
            std::printf(" | latency min/mean/max %.2f/%.2f/%.2f ms", latency.min_ms,
                        latency.mean_ms(), latency.max_ms);
          }
#ifdef PERCEPTION_TRACE_POOL
          std::printf(" | pool %u/%u held peak=%u starved=%lu hold mean/max %.1f/%lu us",
                      source.held(), config.pipeline.ingress_depth, source.held_peak(),
                      source.starved(), source.hold_mean_us(), source.hold_max_us());
#endif
          std::printf("\n");
          latency.reset();
        }
      }
    } catch (...) {
      source.stop();
      upload.stop();
      stop_helpers();
      throw;
    }

    source.stop();
    upload.stop();
    stop_helpers();
    cudaStreamSynchronize(consumer);

    std::printf("\ndelivered=%lu uploaded=%lu consumed=%lu incomplete=%lu foreign=%lu "
                "timeouts=%lu stalls=%lu failed=%lu\n",
                source.delivered(), upload.uploaded(), consumed, source.incomplete(),
                source.foreign(), source.timeouts(), device_ring.write_stalls(), upload.failed());
#ifdef PERCEPTION_TRACE_POOL
    std::printf("pool: depth=%u peak=%u still-held=%u starved=%lu reclaimed=%lu "
                "hold mean/max %.1f/%lu us\n",
                config.pipeline.ingress_depth, source.held_peak(), source.held(),
                source.starved(), source.reclaimed(), source.hold_mean_us(),
                source.hold_max_us());
#endif
  } catch (const std::exception& e) {
    std::printf("FAILED: %s\n", e.what());
    status = 1;
  }

  cameras.Clear();
  system->ReleaseInstance();
  return status;
}
