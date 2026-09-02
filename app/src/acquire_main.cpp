#include <cuda_runtime.h>
#include <pthread.h>

#include <array>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "acquire_source.hpp"
#include "action_sync_check.hpp"
#include "app_config.hpp"
#include "cuda_util.hpp"
#include "device_ring_buffer.hpp"
#include "ess_engine.hpp"
#include "ess_viewer_consumer.hpp"
#include "host_ingress_ring.hpp"
#include "latency_probe.hpp"
#include "report.hpp"
#include "ring_frame_sink.hpp"
#include "headless_yolo_consumer.hpp"
#include "ring_pair_consumer.hpp"
#include "transforms/debayer.hpp"
#include "upload_stage.hpp"
#include "viewer_consumer.hpp"
#include "yolo_viewer_consumer.hpp"

namespace {

constexpr uint32_t kMaxStreams = 2;

std::atomic<bool> g_stop{false};

std::array<std::atomic<perception::DeviceRingBuffer*>, kMaxStreams> g_rings{};
sigset_t stop_signals;

constexpr uint32_t kPipelineConsumerId = 0;
constexpr uint32_t kSecondaryConsumerId = 1;
constexpr uint32_t kRingPairConsumerId = 2;

using perception::cuda_error_check;

void wake_all_rings() {
  for (auto& slot : g_rings) {
    if (perception::DeviceRingBuffer* ring = slot.load(std::memory_order_acquire)) {
      ring->wake_all();
    }
  }
}

// One camera's path from the wire to the device ring.
struct StreamPipeline {
  std::string role;
  perception::FrameSource* source = nullptr;

  perception::LatencyProbe probe;
  std::unique_ptr<perception::HostIngressRing> ingress;
  std::unique_ptr<perception::RingFrameSink> sink;
  std::unique_ptr<perception::DeviceRingBuffer> device_ring;
  std::unique_ptr<perception::UploadStage> upload;
  perception::ActionSyncChecker action_sync_checker;
};

}  // namespace

int main(int argc, char** argv) {
  sigemptyset(&stop_signals);
  sigaddset(&stop_signals, SIGINT);
  if (pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr) != 0) {
    std::printf("FAILED: could not block SIGINT\n");
    return 1;
  }

  std::thread signal_relay([] {
    int signal_number = 0;
    sigwait(&stop_signals, &signal_number);
    g_stop.store(true, std::memory_order_relaxed);
    wake_all_rings();
  });
  struct SignalRelayGuard {
    std::thread& t;
    ~SignalRelayGuard() {
      if (t.joinable()) {
        pthread_kill(t.native_handle(), SIGINT);
        t.join();
      }
    }
  } signal_relay_guard{signal_relay};

  const std::string config_path = argc > 1 ? argv[1] : perception::default_config_path();

  int status = 0;
  try {
    const perception::AppConfig config = perception::load_app_config(config_path);
    std::printf("config: %s\n", config_path.c_str());
    std::printf("viewer: %s\n", perception::to_string(config.viewer_mode));

    // ---- source: live cameras or a recording ----
    std::unique_ptr<perception::AcquireSource> acquire_source =
        perception::make_acquire_source(config);
    const uint32_t stream_count = acquire_source->stream_count();
    if (stream_count == 0 || stream_count > kMaxStreams) {
      throw std::runtime_error("source opened " + std::to_string(stream_count) +
                               " streams; this app handles one or two");
    }
    for (uint32_t s = 0; s < stream_count; ++s) {
      std::printf("source: %s -- %s\n", perception::acquire_source_kind(),
                  acquire_source->describe(s).c_str());
    }

    const bool stereo_enabled = config.stereo.enabled && stream_count == 2;
    if (config.stereo.enabled && !stereo_enabled) {
      std::printf(
          "stereo: disabled -- the config asks for pairing but this source opened %u stream(s).\n"
          "        A recording build replays one stream by design; see source_recording.cpp.\n",
          stream_count);
    }

    const perception::CameraGeometry& geometry = acquire_source->source(0).geometry();
    const perception::ImageDesc desc = perception::to_image_desc(geometry);
    std::printf("frames: %ux%u stride=%u %s, %zu bytes/frame, %zu bytes/buffer, min %u buffers\n",
                geometry.width, geometry.height, geometry.stride_bytes,
                geometry.pixel_format.c_str(), geometry.frame_bytes, geometry.buffer_bytes,
                acquire_source->source(0).min_slot_count());

    if (config.have_calibration) {
      std::printf("%s\n", config.calibration.summary().c_str());
      if (config.calibration.size.width != geometry.width ||
          config.calibration.size.height != geometry.height) {
        throw std::runtime_error(
            "calibration was solved at " + std::to_string(config.calibration.size.width) + "x" +
            std::to_string(config.calibration.size.height) + " but the cameras came up " +
            std::to_string(geometry.width) + "x" + std::to_string(geometry.height) +
            " -- re-run the calibration at this geometry, or set camera.features' Width/Height "
            "back to what it was calibrated at");
      }
    }

    // Read right after Init(): GevIEEE1588Status won't be Slave yet even on a
    // healthy setup (BMC takes a few Announce intervals to settle), so this is
    // "does the node exist / feature applied", not "is it locked" -- Reporter's
    // periodic line is what confirms lock. Empty for a source with no clock.
    const std::string ptp_status = acquire_source->source(0).ptp_status();
    if (ptp_status.rfind("recorded:", 0) == 0) {
      std::printf("ptp: %s (what the rig reported when this was recorded)\n",
                  ptp_status.c_str());
    } else if (!ptp_status.empty()) {
      for (uint32_t s = 0; s < stream_count; ++s) {
        std::printf("ptp: %s status=%s (see spinnaker/README.md if this doesn't reach Slave)\n",
                    acquire_source->describe(s).c_str(),
                    acquire_source->source(s).ptp_status().c_str());
      }
    }

    // ---- device: H2D + debayer -> device ring, one chain per stream ----
    perception::DebayerTransform debayer;
    const perception::ImageDesc display_desc = debayer.output_desc(desc);
    std::printf("transform: %s -> %ux%u stride=%u RGBA8, %zu bytes/frame\n", debayer.name(),
                display_desc.width, display_desc.height, display_desc.stride_bytes,
                display_desc.bytes());

    perception::DeviceRingTextureDesc display_texture;
    display_texture.width = display_desc.width;
    display_texture.height = display_desc.height;
    display_texture.pitch_bytes = display_desc.stride_bytes;

    std::vector<std::unique_ptr<StreamPipeline>> streams;
    for (uint32_t s = 0; s < stream_count; ++s) {
      auto stream = std::make_unique<StreamPipeline>();
      stream->role = s < config.streams.size() ? config.streams[s].role : "cam" + std::to_string(s);
      stream->source = &acquire_source->source(s);
      stream->action_sync_checker.label = stream->role;

      stream->ingress = std::make_unique<perception::HostIngressRing>(
          config.pipeline.ingress_depth, geometry.buffer_bytes, config.pipeline.device_id,
          perception::FillMode::External);
      stream->sink =
          std::make_unique<perception::RingFrameSink>(*stream->ingress, &stream->probe);

      stream->device_ring = std::make_unique<perception::DeviceRingBuffer>(
          config.pipeline.device_depth, display_desc.bytes(), config.pipeline.reuse_wait,
          config.pipeline.write_policy, config.pipeline.max_consumers, config.pipeline.device_id,
          display_texture);
      g_rings[s].store(stream->device_ring.get(), std::memory_order_release);

      stream->upload = std::make_unique<perception::UploadStage>(
          *stream->ingress, *stream->device_ring, debayer, desc, config.upload);

      streams.push_back(std::move(stream));
    }

    struct RingUnregister {
      std::thread& t;
      ~RingUnregister() {
        if (t.joinable()) {
          pthread_kill(t.native_handle(), SIGINT);
          t.join();
        }
        for (auto& slot : g_rings) slot.store(nullptr, std::memory_order_release);
      }
    } ring_unregister{signal_relay};

    // Everything downstream of the rings hangs off the reference stream
    StreamPipeline& reference = *streams[0];

    perception::YoloConfig yolo_config = config.yolo;
    yolo_config.engine_path = perception::resolve_next_to_exe(yolo_config.engine_path);

    std::unique_ptr<perception::ViewerConsumer> camera_viewer;
    std::unique_ptr<perception::YoloViewerConsumer> yolo_viewer;
    std::unique_ptr<perception::HeadlessYoloConsumer> headless_yolo;

    switch (config.viewer_mode) {
      case perception::ViewerMode::Camera:
        camera_viewer = std::make_unique<perception::ViewerConsumer>(
            *reference.device_ring, reference.probe, display_desc, config.display,
            kSecondaryConsumerId, config.pipeline.device_id);
        break;
      case perception::ViewerMode::Yolo:
        yolo_viewer = std::make_unique<perception::YoloViewerConsumer>(
            *reference.device_ring, reference.probe, display_desc, config.display, yolo_config,
            kSecondaryConsumerId, config.pipeline.device_id);
        break;
      case perception::ViewerMode::Headless:
        headless_yolo = std::make_unique<perception::HeadlessYoloConsumer>(
            *reference.device_ring, display_desc, yolo_config,
            perception::HeadlessYoloConsumer::OutputLocation::Host, kSecondaryConsumerId,
            config.pipeline.device_id);
        break;
      case perception::ViewerMode::Ess:
        // Built below: it drives the pair consumer, which does not exist yet.
        break;
    }

    const bool ess_in_viewer = config.viewer_mode == perception::ViewerMode::Ess;
    std::unique_ptr<perception::EssEngine> ess;
    if (config.ess.enabled && stereo_enabled && !ess_in_viewer) {
      perception::EssEngine::Config ess_config;
      ess_config.engine_path = perception::resolve_next_to_exe(config.ess.engine_path);
      ess_config.plugin_path = perception::resolve_next_to_exe(config.ess.plugin_path);
      ess_config.normalization = config.ess.normalization;
      ess_config.conf_threshold = config.ess.conf_threshold;
      ess_config.device_id = config.pipeline.device_id;
      ess = std::make_unique<perception::EssEngine>(display_desc, config.calibration,
                                                    std::move(ess_config));
    } else if (config.ess.enabled && !stereo_enabled) {
      std::printf("ess: disabled -- disparity needs a pair, and stereo pairing is off\n");
    }

    std::unique_ptr<perception::RingPairConsumer> stereo;
    // After `stereo`, so its thread -- which this one drives -- is joined
    // before the consumer it is driving goes away.
    std::unique_ptr<perception::EssViewerConsumer> ess_viewer;
    if (stereo_enabled) {
      const uint32_t ref_index = config.stereo.reference_stream;
      const uint32_t other_index = ref_index == 0 ? 1u : 0u;

      stereo = std::make_unique<perception::RingPairConsumer>(
          *streams[ref_index]->device_ring, *streams[other_index]->device_ring,
          config.stereo.consumer, kRingPairConsumerId, config.pipeline.device_id);

      if (ess) {
        stereo->set_pair_callback([&ess, ref_index](perception::ReadLease& reference,
                                                    perception::ReadLease& other, int64_t skew_ns,
                                                    uint64_t pair_id, cudaStream_t stream) {
        perception::ReadLease& left = ref_index == 0 ? reference : other;
        perception::ReadLease& right = ref_index == 0 ? other : reference;

        ess->preprocess(left.texture(), right.texture(), stream);

        left.drop_hold();
        right.drop_hold();

        ess->infer(stream);

        if (pair_id % 60 != 0) return;

        perception::EssEngine::FrameTiming timing;
        perception::EssEngine::Sample sample;
        const bool have_timing = ess->last_timing(timing);
        const bool have_sample = ess->last_sample(sample);
        std::printf(
            "  ess: pairs=%lu skew=%ldus pre=%.2fms infer=%.2fms total=%.2fms "
            "d(%u,%u)=%.2fpx z=%.2fm conf=%.2f%s\n",
            static_cast<unsigned long>(pair_id + 1), static_cast<long>(skew_ns / 1000),
            have_timing ? timing.preprocess_ms : -1.0, have_timing ? timing.inference_ms : -1.0,
            have_timing ? timing.total_ms : -1.0, sample.x, sample.y,
            have_sample ? sample.disparity_px : -1.0f, have_sample ? sample.depth_m : -1.0,
            have_sample ? sample.confidence : -1.0f,
            have_sample && !sample.trusted ? " (untrusted)" : "");
        });
      }

      // The disparity window sets its own pair callback and drives step()
      // itself, so the pair consumer's thread stays unstarted below.
      if (ess_in_viewer && config.ess.enabled) {
        perception::EssConfig ess_view_config = config.ess;
        ess_view_config.engine_path = perception::resolve_next_to_exe(ess_view_config.engine_path);
        ess_view_config.plugin_path = perception::resolve_next_to_exe(ess_view_config.plugin_path);
        ess_viewer = std::make_unique<perception::EssViewerConsumer>(
            *stereo, ref_index, reference.probe, display_desc, config.calibration, config.display,
            ess_view_config, config.pipeline.device_id);
      }

      std::printf(
          "stereo: pairing %s (reference) against %s at %luus, %u retry x %ldms, calibration %s\n",
          streams[ref_index]->role.c_str(), streams[other_index]->role.c_str(),
          static_cast<unsigned long>(config.stereo.consumer.tolerance_ns / 1000),
          config.stereo.consumer.retry_attempts,
          static_cast<long>(config.stereo.consumer.retry_wait.count()),
          config.have_calibration ? "loaded" : "none");
      if (!ess_viewer) stereo->start();
    }

    // Closed only ever means "the window the user had open just went away";
    // headless has no window, so it never ends the run on its own.
    auto secondary_closed = [&] {
      if (camera_viewer) return camera_viewer->closed();
      if (yolo_viewer) return yolo_viewer->closed();
      if (ess_viewer) return ess_viewer->closed();
      return false;
    };

    if (camera_viewer) camera_viewer->start();
    if (yolo_viewer) yolo_viewer->start();
    if (headless_yolo) headless_yolo->start();
    if (ess_viewer) ess_viewer->start();

    auto stop_helpers = [&] {
      if (ess_viewer) ess_viewer->stop();
      if (stereo) stereo->stop();
      if (camera_viewer) camera_viewer->stop();
      if (yolo_viewer) yolo_viewer->stop();
      if (headless_yolo) headless_yolo->stop();
      wake_all_rings();
      if (signal_relay.joinable()) {
        pthread_kill(signal_relay.native_handle(), SIGINT);
        signal_relay.join();
      }
    };

    perception::Reporter reporter(*reference.source, *reference.upload, *reference.device_ring,
                                  reference.probe, config, ptp_status);

    auto any_finished = [&] {
      for (const auto& stream : streams) {
        if (stream->source->finished()) return true;
      }
      return false;
    };

    uint64_t consumed = 0;
    uint32_t last_slot = ~0u;
    uint64_t last_seq = ~0ull;

    perception::CudaStream consumer;
    try {
      for (auto& stream : streams) {
        stream->source->set_finished_callback([] { wake_all_rings(); });
        stream->source->start(*stream->sink);
        stream->upload->start();
      }

      std::vector<perception::ActionSyncChecker*> checkers;
      for (auto& stream : streams) checkers.push_back(&stream->action_sync_checker);
      acquire_source->arm_action_sync(config.action_sync, checkers);

      while (!g_stop.load(std::memory_order_relaxed) && !secondary_closed() && !any_finished()) {
        const uint64_t seen = reference.device_ring->wait_seq();

        perception::FramePeek peek;
        if (!reference.device_ring->view_latest_inplace(peek) ||
            (peek.slot == last_slot && peek.slot_seq == last_seq)) {
          reference.device_ring->wait_for_publish(seen);
          continue;
        }

        perception::ReadLease lease =
            reference.device_ring->lease_latest(kPipelineConsumerId, consumer);
        if (!lease.valid()) continue;
        last_slot = lease.slot();
        last_seq = lease.seq();

        cuda_error_check(cudaStreamWaitEvent(consumer, lease.data_ready_event(), 0),
                         "cudaStreamWaitEvent");

        ++consumed;
        for (auto& stream : streams) {
          if (stream.get() == &reference) stream->action_sync_checker.observe(lease.timestamp_ns());
        }
        reporter.observe(consumed, lease);

        if (consumed % 60 == 0) {
          const std::string trig = acquire_source->trigger_health_line();
          if (stereo && !trig.empty()) {
            std::printf("  %s | %s\n", stereo->health_line().c_str(), trig.c_str());
          } else if (stereo) {
            std::printf("  %s\n", stereo->health_line().c_str());
          } else if (!trig.empty()) {
            std::printf("  %s\n", trig.c_str());
          }
        }
      }
    } catch (...) {
      // Before the sources: the trigger thread touches the SDK they own.
      acquire_source->stop_action_sync();
      for (auto& stream : streams) {
        stream->source->stop();
        stream->upload->stop();
      }
      stop_helpers();
      throw;
    }

    acquire_source->stop_action_sync();
    for (auto& stream : streams) {
      stream->source->stop();
      stream->upload->stop();
    }
    stop_helpers();
    cudaStreamSynchronize(consumer);

    reporter.print_summary(consumed);
    if (stereo) std::printf("%s\n", stereo->health_line().c_str());
    {
      const std::string trig = acquire_source->trigger_health_line();
      if (!trig.empty()) std::printf("%s\n", trig.c_str());
    }

    for (auto& stream : streams) {
      if (stream->source->failed()) {
        std::printf("acquisition failed on %s: %s\n", stream->role.c_str(),
                    stream->source->failure().c_str());
        status = 1;
      }
    }
  } catch (const std::exception& e) {
    std::printf("FAILED: %s\n", e.what());
    status = 1;
  }

  return status;
}
