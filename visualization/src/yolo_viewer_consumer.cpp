#include "yolo_viewer_consumer.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <thread>

#include "cuda_util.hpp"
#include "gl_viewer.hpp"
#include "yolo_engine.hpp"

namespace perception {
namespace {

constexpr double kPollSeconds = 0.005;
constexpr uint64_t kReportEvery = 60;

YoloEngine::Config to_engine_config(const YoloConfig& config, int device_id) {
  YoloEngine::Config engine_config;
  engine_config.engine_path = config.engine_path;
  engine_config.model_width = config.model_width;
  engine_config.model_height = config.model_height;
  engine_config.conf_threshold = config.conf_threshold;
  engine_config.device_id = device_id;
  return engine_config;
}

}  // namespace

struct YoloViewerConsumer::Impl {
  DeviceRingBuffer* ring;
  const LatencyProbe* probe;
  ImageDesc desc;
  DisplayConfig display_config;
  YoloConfig yolo_config;
  uint32_t consumer_id;
  int device_id;

  std::thread thread;
  std::atomic<bool> running{false};
  std::atomic<bool> closed{false};
  double last_process_ms = -1.0;
  uint64_t last_report_ns = 0;

  Impl(DeviceRingBuffer& ring_ref, const LatencyProbe& probe_ref, const ImageDesc& desc_in,
      const DisplayConfig& display_config_in, const YoloConfig& yolo_config_in,
      uint32_t consumer_id_in, int device_id_in)
      : ring(&ring_ref),
        probe(&probe_ref),
        desc(desc_in),
        display_config(display_config_in),
        yolo_config(yolo_config_in),
        consumer_id(consumer_id_in),
        device_id(device_id_in) {
    if (consumer_id >= ring->max_consumers()) {
      throw std::runtime_error(
          "YoloViewerConsumer: consumer id is outside the ring's max_consumers");
    }
  }

  void run() {
    GlViewer::Config view_config;
    view_config.window_width = display_config.window_width;
    view_config.window_height = display_config.window_height;
    view_config.vsync = display_config.vsync;
    view_config.latency_scale_ms = display_config.latency_scale_ms;

    // Constructed here and nowhere else: the GL context is thread-current and
    // the CUDA-GL interop calls need it bound.
    std::unique_ptr<GlViewer> viewer;
    try {
      viewer = std::make_unique<GlViewer>(desc, view_config);
      std::printf("display: %ux%u vsync=%s scale=%.0fms (esc/q to quit)\n",
                 display_config.window_width, display_config.window_height,
                 display_config.vsync ? "on" : "off", display_config.latency_scale_ms);
    } catch (const std::exception& e) {
      std::printf("display: disabled (%s)\n", e.what());
      return;
    }

    if (cudaSetDevice(device_id) != cudaSuccess) {
      std::printf("display: disabled (cudaSetDevice failed on the yolo viewer thread)\n");
      return;
    }

    // An engine that fails to load (bad path, version mismatch) does not
    // take the window down with it
    std::unique_ptr<YoloEngine> engine;
    try {
      engine = std::make_unique<YoloEngine>(desc, to_engine_config(yolo_config, device_id));
    } catch (const std::exception& e) {
      std::printf("yolo: disabled (%s)\n", e.what());
    }

    CudaStream stream;
    uint32_t last_slot = ~0u;
    uint64_t last_seq = ~0ull;
    last_report_ns = LatencyProbe::host_now_ns();

    try {
      while (running.load(std::memory_order_relaxed)) {
        viewer->poll_wait(kPollSeconds);
        if (viewer->should_close()) break;

        FramePeek peek;
        if (!ring->view_latest_inplace(peek) ||
            (peek.slot == last_slot && peek.slot_seq == last_seq)) {
          continue;
        }

        ReadLease lease = ring->lease_latest(consumer_id, stream);
        if (!lease.valid()) continue;
        last_slot = lease.slot();
        last_seq = lease.seq();

        cuda_error_check(cudaStreamWaitEvent(stream, lease.data_ready_event(), 0),
                         "YoloViewerConsumer: cudaStreamWaitEvent");

        double latency_ms = -1.0;
        int64_t age_ns = 0;
        if (probe->age_ns(lease.timestamp_ns(), age_ns)) {
          latency_ms = static_cast<double>(age_ns) * 1e-6;
        }

        if (engine) {
          double process_ms = 0.0;
          if (engine->process_ms(process_ms)) last_process_ms = process_ms;

          engine->enqueue(lease.texture(), stream);
          viewer->present_gpu_boxes(lease.data(), stream, latency_ms,
                                   [&](cudaSurfaceObject_t surface) {
                                     engine->draw_into(surface, stream);
                                   });
        } else {
          viewer->present(lease.data(), stream, latency_ms);
        }

        if (viewer->presented() % kReportEvery == 0) {
          const uint64_t now_ns = LatencyProbe::host_now_ns();
          const double elapsed_s = static_cast<double>(now_ns - last_report_ns) * 1e-9;
          const double fps = elapsed_s > 0.0 ? static_cast<double>(kReportEvery) / elapsed_s : 0.0;
          last_report_ns = now_ns;

          std::printf("display: fps=%.2f", fps);
          if (engine) std::printf(" | yolo process=%.2fms", last_process_ms);
          std::printf("\n");
        }
        // Lease goes out of scope and destructor returns the slot
      }
    } catch (const std::exception& e) {
      std::printf("display: stopped (%s)\n", e.what());
    }

    closed.store(true, std::memory_order_relaxed);
    ring->wake_all();
    cudaStreamSynchronize(stream);
  }
};

YoloViewerConsumer::YoloViewerConsumer(DeviceRingBuffer& ring, const LatencyProbe& probe,
                                      const ImageDesc& source_desc,
                                      const DisplayConfig& display_config,
                                      const YoloConfig& yolo_config, uint32_t consumer_id,
                                      int device_id)
    : impl_(std::make_unique<Impl>(ring, probe, source_desc, display_config, yolo_config,
                                  consumer_id, device_id)) {}

YoloViewerConsumer::~YoloViewerConsumer() { stop(); }

void YoloViewerConsumer::start() {
  if (impl_->running.exchange(true)) return;
  impl_->thread = std::thread([this] { impl_->run(); });
}

void YoloViewerConsumer::stop() {
  impl_->running.store(false, std::memory_order_relaxed);
  if (impl_->thread.joinable()) impl_->thread.join();
}

bool YoloViewerConsumer::closed() const { return impl_->closed.load(std::memory_order_relaxed); }

}  // namespace perception
