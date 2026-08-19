#include <Spinnaker.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

#include "action_sync.hpp"
#include "app_config.hpp"
#include "cuda_util.hpp"
#include "device_ring_buffer.hpp"
#include "host_ingress_ring.hpp"
#include "latency_probe.hpp"
#include "report.hpp"
#include "ring_frame_sink.hpp"
#include "spinnaker_source.hpp"
#include "transforms/debayer.hpp"
#include "upload_stage.hpp"
#include "viewer_consumer.hpp"

namespace {

std::atomic<bool> g_stop{false};
// Set once device_ring exists, so the signal thread (armed before it does)
// has something to wake once there is something worth waking.
std::atomic<perception::DeviceRingBuffer*> g_ring{nullptr};
sigset_t stop_signals;

constexpr uint32_t kPipelineConsumerId = 0;
constexpr uint32_t kViewerConsumerId = 1;

using perception::cuda_error_check;

// Resolved against the binary's own directory, not the caller's cwd, so
// `./build/bin/acquire` works the same from the project root, from
// build/bin, or anywhere else -- matching where CMake's POST_BUILD step
// actually links config/ (next to the binary, not next to the source).
std::string default_config_path() {
  std::error_code ec;
  const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  const std::filesystem::path base = ec ? std::filesystem::current_path() : exe.parent_path();
  return (base / "config" / "acquire.yaml").string();
}

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
    if (perception::DeviceRingBuffer* ring = g_ring.load(std::memory_order_acquire)) {
      ring->wake_all();
    }
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

  const std::string config_path = argc > 1 ? argv[1] : default_config_path();

  Spinnaker::SystemPtr system = Spinnaker::System::GetInstance();
  Spinnaker::CameraList cameras = system->GetCameras();

  int status = 0;
  try {
    const perception::AppConfig config = perception::load_app_config(config_path);
    std::printf("config: %s\n", config_path.c_str());

    // ---- camera: pixels land straight in the sink's user buffers ----
    perception::SpinnakerSource source(
        perception::SpinnakerSource::select(cameras, config.camera.serial), config.camera);

    const perception::CameraGeometry& geometry = source.geometry();
    const perception::ImageDesc desc = perception::to_image_desc(geometry);
    std::printf("camera: %ux%u stride=%u %s, %zu bytes/frame, %zu bytes/buffer, min %u buffers\n",
                geometry.width, geometry.height, geometry.stride_bytes,
                geometry.pixel_format.c_str(), geometry.frame_bytes, geometry.buffer_bytes,
                source.min_slot_count());

    // Read right after Init(): GevIEEE1588Status won't be Slave yet even on a
    // healthy setup (BMC takes a few Announce intervals to settle), so this is
    // "does the node exist / feature applied", not "is it locked" -- Reporter's
    // periodic line is what confirms lock.
    const std::string ptp_status = source.ptp_status();
    if (!ptp_status.empty()) {
      std::printf("ptp: status=%s (see spinnaker/README.md if this doesn't reach Slave)\n",
                  ptp_status.c_str());
    }

    perception::LatencyProbe probe;

    // ---- host: camera buffer -> pinned ingress ring ----
    perception::HostIngressRing ingress(config.pipeline.ingress_depth, geometry.buffer_bytes,
                                        config.pipeline.device_id, perception::FillMode::External);
    perception::RingFrameSink sink(ingress, &probe);

    // ---- device: H2D + debayer -> device ring ----
    perception::DebayerTransform debayer;
    const perception::ImageDesc display_desc = debayer.output_desc(desc);
    std::printf("transform: %s -> %ux%u stride=%u RGBA8, %zu bytes/frame\n", debayer.name(),
                display_desc.width, display_desc.height, display_desc.stride_bytes,
                display_desc.bytes());

    perception::DeviceRingBuffer device_ring(
        config.pipeline.device_depth, display_desc.bytes(), config.pipeline.reuse_wait,
        config.pipeline.write_policy, config.pipeline.max_consumers, config.pipeline.device_id);
    g_ring.store(&device_ring, std::memory_order_release);
    
    struct RingUnregister {
      std::thread& t;
      ~RingUnregister() {
        if (t.joinable()) {
          pthread_kill(t.native_handle(), SIGINT);
          t.join();
        }
        g_ring.store(nullptr, std::memory_order_release);
      }
    } ring_unregister{signal_relay};

    // With a transform the H2D lands in scratch and the debayer writes the
    // output slot, so the camera's buffer is released one step earlier than on
    // the upload-only path.
    perception::UploadStage upload(ingress, device_ring, debayer, desc, config.upload);

    // ---- device ring -> display, the debug consumer ----
    perception::ViewerConsumer viewer(device_ring, probe, display_desc, config.display,
                                      kViewerConsumerId, config.pipeline.device_id);
    viewer.start();

    auto stop_helpers = [&] {
      viewer.stop();
      device_ring.wake_all();
      if (signal_relay.joinable()) {
        pthread_kill(signal_relay.native_handle(), SIGINT);
        signal_relay.join();
      }
    };

    perception::Reporter reporter(source, upload, device_ring, probe, config, ptp_status);
    perception::ActionSyncChecker action_sync_checker;

    uint64_t consumed = 0;
    uint32_t last_slot = ~0u;
    uint64_t last_seq = ~0ull;

    perception::CudaStream consumer;
    try {
      source.start(sink);
      upload.start();

      if (config.action_sync.enabled) {
        perception::arm_action_sync(system, source, config.action_sync, action_sync_checker);
      }

      // ---- pipeline consumer: lease the latest frame, order the CUDA
      // consumer stream behind it, and report ----
      while (!g_stop.load(std::memory_order_relaxed) && !viewer.closed()) {
        const uint64_t seen = device_ring.wait_seq();

        perception::FramePeek peek;
        if (!device_ring.view_latest_inplace(peek) ||
            (peek.slot == last_slot && peek.slot_seq == last_seq)) {
          device_ring.wait_for_publish(seen);
          continue;
        }

        perception::ReadLease lease = device_ring.lease_latest(kPipelineConsumerId, consumer);
        if (!lease.valid()) continue;
        last_slot = lease.slot();
        last_seq = lease.seq();

        cuda_error_check(cudaStreamWaitEvent(consumer, lease.data_ready_event(), 0),
                         "cudaStreamWaitEvent");

        ++consumed;
        action_sync_checker.observe(lease.timestamp_ns());
        reporter.observe(consumed, lease);
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

    reporter.print_summary(consumed);
  } catch (const std::exception& e) {
    std::printf("FAILED: %s\n", e.what());
    status = 1;
  }

  cameras.Clear();
  system->ReleaseInstance();
  return status;
}
