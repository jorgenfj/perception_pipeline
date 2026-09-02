#include "ess_viewer_consumer.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <memory>
#include <thread>

#include "cuda_util.hpp"
#include "ess_engine.hpp"
#include "gl_viewer.hpp"

namespace perception {
namespace {

constexpr double kPollSeconds = 0.005;
constexpr uint64_t kReportEvery = 60;

EssEngine::Config to_engine_config(const EssConfig& config, int device_id) {
  EssEngine::Config engine_config;
  engine_config.engine_path = config.engine_path;
  engine_config.plugin_path = config.plugin_path;
  engine_config.normalization = config.normalization;
  engine_config.conf_threshold = config.conf_threshold;
  engine_config.colormap = config.colormap;
  engine_config.display_min_disparity = config.display_min_disparity;
  engine_config.display_max_disparity = config.display_max_disparity;
  engine_config.device_id = device_id;
  return engine_config;
}

}  // namespace

struct EssViewerConsumer::Impl {
  RingPairConsumer* pairs;
  uint32_t reference_stream;
  const LatencyProbe* probe;
  ImageDesc desc;
  utils::StereoCalibration calibration;
  DisplayConfig display_config;
  EssConfig ess_config;
  int device_id;

  std::thread thread;
  std::atomic<bool> running{false};
  std::atomic<bool> closed{false};

  Impl(RingPairConsumer& pairs_ref, uint32_t reference_stream_in, const LatencyProbe& probe_ref,
      const ImageDesc& desc_in, const utils::StereoCalibration& calibration_in,
      const DisplayConfig& display_config_in, const EssConfig& ess_config_in, int device_id_in)
      : pairs(&pairs_ref),
        reference_stream(reference_stream_in),
        probe(&probe_ref),
        desc(desc_in),
        calibration(calibration_in),
        display_config(display_config_in),
        ess_config(ess_config_in),
        device_id(device_id_in) {}

  void run() {
    if (cudaSetDevice(device_id) != cudaSuccess) {
      std::printf("ess viewer: disabled (cudaSetDevice failed on the viewer thread)\n");
      closed.store(true, std::memory_order_relaxed);
      return;
    }

    // The window is the network's own grid, not the camera's: upscaling a
    // disparity map only invents edges that the matcher never produced.
    const ImageDesc disparity_desc =
        packed_desc(kEssFullWidth, kEssFullHeight, PixelFormat::RGBA8);

    GlViewer::Config view_config;
    view_config.window_width = display_config.window_width;
    view_config.window_height = display_config.window_height;
    view_config.vsync = display_config.vsync;
    view_config.latency_scale_ms = display_config.latency_scale_ms;

    // Constructed here and nowhere else: the GL context is thread-current and
    // the CUDA-GL interop calls need it bound.
    std::unique_ptr<GlViewer> viewer;
    std::unique_ptr<EssEngine> engine;
    try {
      viewer = std::make_unique<GlViewer>(disparity_desc, view_config);
      engine = std::make_unique<EssEngine>(desc, calibration,
                                          to_engine_config(ess_config, device_id));
      std::printf("ess viewer: %ux%u disparity, vsync=%s (esc/q to quit)\n", disparity_desc.width,
                 disparity_desc.height, display_config.vsync ? "on" : "off");
    } catch (const std::exception& e) {
      std::printf("ess viewer: disabled (%s)\n", e.what());
      closed.store(true, std::memory_order_relaxed);
      return;
    }

    CudaStream stream;
    uint64_t presented_at_report = 0;
    uint64_t last_report_ns = LatencyProbe::host_now_ns();

    // Runs on this thread, inside step(), with both leases held. Everything it
    // enqueues goes on `stream`, which step() has already ordered behind both
    // frames -- and nothing in it waits on the GPU, so the leases are not held
    // across a stall.
    pairs->set_pair_callback([&](ReadLease& reference, ReadLease& other, int64_t /*skew_ns*/,
                                 uint64_t /*pair_id*/, cudaStream_t s) {
      // The pair consumer hands its own reference back first, and that is
      // streams[reference_stream]; the calibration's cameras[0] is always
      // streams[0]. Swapping the two is not something the engine can see -- it
      // is a disparity of the wrong sign that still looks like one.
      ReadLease& left = reference_stream == 0 ? reference : other;
      ReadLease& right = reference_stream == 0 ? other : reference;

      const uint64_t reference_timestamp_ns = reference.timestamp_ns();
      engine->preprocess(left.texture(), right.texture(), s);
      
      left.drop_hold();
      right.drop_hold();

      engine->infer(s);

      double latency_ms = -1.0;
      int64_t age_ns = 0;
      if (probe->age_ns(reference_timestamp_ns, age_ns)) {
        latency_ms = static_cast<double>(age_ns) * 1e-6;
      }

      // No base frame to copy: draw_into writes every texel of the surface.
      viewer->present_gpu(s, latency_ms,
                         [&](cudaSurfaceObject_t surface) { engine->draw_into(surface, s); });
    });

    try {
      while (running.load(std::memory_order_relaxed)) {
        viewer->poll_wait(kPollSeconds);
        if (viewer->should_close()) break;

        // False just means no new reference frame, or none that could be
        // paired; the pair consumer's own counters say which.
        if (!pairs->step(stream)) continue;

        if (viewer->presented() - presented_at_report < kReportEvery) continue;
        const uint64_t now_ns = LatencyProbe::host_now_ns();
        const double elapsed_s = static_cast<double>(now_ns - last_report_ns) * 1e-9;
        const double fps =
            elapsed_s > 0.0 ? static_cast<double>(viewer->presented() - presented_at_report) /
                                  elapsed_s
                            : 0.0;
        presented_at_report = viewer->presented();
        last_report_ns = now_ns;

        EssEngine::FrameTiming timing;
        EssEngine::Sample sample;
        const bool have_timing = engine->last_timing(timing);
        const bool have_sample = engine->last_sample(sample);
        std::printf("ess viewer: fps=%.2f | %s", fps, pairs->health_line().c_str());
        if (have_timing) {
          std::printf(" | pre=%.1fms infer=%.1fms total=%.1fms", timing.preprocess_ms,
                     timing.inference_ms, timing.total_ms);
        }
        if (have_sample) {
          std::printf(" | d(%u,%u)=%.2fpx z=%.2fm%s", sample.x, sample.y, sample.disparity_px,
                     sample.depth_m, sample.trusted ? "" : " (untrusted)");
        }
        std::printf("\n");
      }
    } catch (const std::exception& e) {
      std::printf("ess viewer: stopped (%s)\n", e.what());
    }

    // Before the viewer and engine this callback captured go out of scope.
    pairs->set_pair_callback(nullptr);
    closed.store(true, std::memory_order_relaxed);
    cudaStreamSynchronize(stream);
  }
};

EssViewerConsumer::EssViewerConsumer(RingPairConsumer& pairs, uint32_t reference_stream,
                                    const LatencyProbe& probe, const ImageDesc& source_desc,
                                    const utils::StereoCalibration& calibration,
                                    const DisplayConfig& display_config,
                                    const EssConfig& ess_config, int device_id)
    : impl_(std::make_unique<Impl>(pairs, reference_stream, probe, source_desc, calibration,
                                  display_config, ess_config, device_id)) {}

EssViewerConsumer::~EssViewerConsumer() { stop(); }

void EssViewerConsumer::start() {
  if (impl_->running.exchange(true)) return;
  impl_->thread = std::thread([this] { impl_->run(); });
}

void EssViewerConsumer::stop() {
  impl_->running.store(false, std::memory_order_relaxed);
  if (impl_->thread.joinable()) impl_->thread.join();
}

bool EssViewerConsumer::closed() const { return impl_->closed.load(std::memory_order_relaxed); }

}  // namespace perception
