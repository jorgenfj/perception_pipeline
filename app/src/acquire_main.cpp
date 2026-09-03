#include <cuda_runtime.h>
#include <pthread.h>

#include <array>
#include <algorithm>
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
#include "host_frame_ring.hpp"
#include "host_ingress_ring.hpp"
#include "latency_probe.hpp"
#include "report.hpp"
#include "ring_frame_sink.hpp"
#include "download_stage.hpp"
#include "mcap_recorder.hpp"
#include "ros_messages.hpp"
#include "headless_yolo_consumer.hpp"
#include "ring_pair_consumer.hpp"
#include "transforms/debayer.hpp"
#include "upload_stage.hpp"
#include "viewer_consumer.hpp"
#include "yolo_viewer_consumer.hpp"

namespace {

constexpr uint32_t kMaxStreams = 2;

// Consumers of a stream's host tap. The recorder is the only one today; anything
// else that wants the bayer frames on the CPU is a second one, and the ring is
// sized consumers + 2.
constexpr uint32_t kTapConsumers = 1;

std::atomic<bool> g_stop{false};

std::array<std::atomic<perception::DeviceRingBuffer*>, kMaxStreams> g_rings{};
sigset_t stop_signals;

constexpr uint32_t kPipelineConsumerId = 0;
constexpr uint32_t kSecondaryConsumerId = 1;
constexpr uint32_t kRingPairConsumerId = 2;

// The one topic name not derived from a camera role.
constexpr const char* kDisparityTopic = "/disparity";

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
  std::unique_ptr<perception::RingFrameSink> ring_sink;

  // The CPU fan-out, built only for a run that has a host-side consumer -- the
  // recorder, today. Null otherwise, and then the acquisition thread does not
  // copy at all.
  std::unique_ptr<perception::HostFrameRing> tap;
  std::thread recorder_thread;

  std::unique_ptr<perception::DeviceRingBuffer> device_ring;
  std::unique_ptr<perception::UploadStage> upload;
  perception::ActionSyncChecker action_sync_checker;

  // Wakes the encoder and waits for it. Idempotent, so the shutdown path can
  // call it and the destructor can call it again.
  void stop_tap() {
    if (tap) tap->stop();
    if (recorder_thread.joinable()) recorder_thread.join();
  }

  // The encoder holds a raw pointer to the tap and to the recorder, and a
  // joinable thread destroyed is a terminate(). stop_helpers() normally gets
  // here first; this covers the paths that throw before it does.
  ~StreamPipeline() { stop_tap(); }
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

  // Parsed rather than indexed, because a silently ignored argument is a run
  // you have to do twice -- stereo_view learned that about --record and says so
  // in its own parser. Anything unrecognised is an error here, not a shrug.
  std::string config_path;
  bool force_record = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--record") {
      force_record = true;
    } else if (arg == "-h" || arg == "--help") {
      std::printf(
          "acquire [options] [config.yaml]\n"
          "\n"
          "  --record    write an MCAP of this run, overriding recording.enabled\n"
          "              in the config. What it captures -- images, disparity --\n"
          "              still comes from the `recording:` section.\n"
          "  -h, --help\n"
          "\n"
          "Without a config path, %s is used.\n",
          perception::default_config_path().c_str());
      return 0;
    } else if (!arg.empty() && arg[0] == '-') {
      std::printf("FAILED: unknown option '%s' (try --help)\n", arg.c_str());
      return 1;
    } else if (config_path.empty()) {
      config_path = arg;
    } else {
      std::printf("FAILED: more than one config path given ('%s' and '%s')\n",
                  config_path.c_str(), arg.c_str());
      return 1;
    }
  }
  if (config_path.empty()) config_path = perception::default_config_path();

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

    // Before the streams: their sinks tap into it. Built here rather than after
    // the engine because the camera channels only need geometry, and the
    // disparity channel's fx/baseline are filled in once the engine exists.
    std::unique_ptr<perception::McapRecorder> recorder;
    // --record turns recording on for this run without editing the file; what
    // it captures still comes from the `recording:` section.
    const bool recording_enabled = config.recording.enabled || force_record;
    if (force_record && !config.recording.enabled) {
      std::printf("recording: enabled by --record\n");
    }
    // load_app_config only validates the recording section when the file turns
    // it on, so --record has to answer for itself: a run that records nothing
    // is one you have to do again.
    if (recording_enabled && config.recording.topics.empty()) {
      std::printf("FAILED: recording is on but recording.topics is empty\n");
      return 1;
    }

    std::vector<std::unique_ptr<StreamPipeline>> streams;
    for (uint32_t s = 0; s < stream_count; ++s) {
      auto stream = std::make_unique<StreamPipeline>();
      stream->role = s < config.streams.size() ? config.streams[s].role : "cam" + std::to_string(s);
      stream->source = &acquire_source->source(s);
      stream->action_sync_checker.label = stream->role;

      stream->ingress = std::make_unique<perception::HostIngressRing>(
          config.pipeline.ingress_depth, geometry.buffer_bytes, config.pipeline.device_id,
          perception::FillMode::External);
      stream->ring_sink = std::make_unique<perception::RingFrameSink>(
          *stream->ingress, &stream->probe, acquire_source->epoch_offset_ns());

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

    // --- recording ------------------------------------------------------------
    // After the engine, because the disparity channel carries the rectified fx
    // and the baseline, and those come from the calibration resized onto the
    // network's grid. Before source->start(), which is the first moment a sink
    // is used, so the decorators below are in place by the time frames arrive.
    std::unique_ptr<perception::DownloadStage> disparity_download;
    if (recording_enabled) {
      // Nested, so adding a recorder knob touches neither this file nor app_config.
      perception::McapRecorder::Config recorder_config = config.recording.recorder;
      // Not a knob: what the sinks already took off, so the file can say so.
      recorder_config.epoch_offset_ns = acquire_source->epoch_offset_ns();
      recorder = std::make_unique<perception::McapRecorder>(recorder_config);

      // Sizes the queues only, so being a factor out costs a queue depth.
      const double topic_rate_hz = config.action_sync.expected_hz;

      // Every topic actually declared, checked against the config below so a
      // name nothing produces stops the run rather than recording silence.
      std::vector<std::string> declared;

      for (uint32_t s = 0; s < stream_count; ++s) {
        const std::string name = "/" + streams[s]->role + "/image_raw";
        if (!config.recording.records(name)) continue;

        // No geometry here: the reservation is the type's own default and each
        // message carries its own desc.
        const auto topic = perception::ros_msg::add_topic<perception::ros_msg::ImageMessage>(
            *recorder, name, topic_rate_hz);
        declared.push_back(name);

        // The tap, and the recorder reading it. Built here rather than with the
        // rest of the stream because this is what knows whether anything wants
        // the frames on the CPU: with no consumer there is no ring and no copy.
        streams[s]->tap = std::make_unique<perception::HostFrameRing>(
            kTapConsumers + 2, geometry.buffer_bytes, desc.width, desc.height);
        const uint32_t consumer = streams[s]->tap->add_consumer("recorder");
        streams[s]->ring_sink->set_tap(streams[s]->tap.get());

        // Everything the encode needs is captured by value -- the topic, the
        // geometry, the frame name -- because this scope is gone long before
        // the thread is. The frame is held for the whole encode and released as
        // the loop turns, which costs a tap slot and nothing on the camera.
        streams[s]->recorder_thread = std::thread(
            [ring = streams[s]->tap.get(), consumer, sink = recorder.get(), topic, desc,
             frame_id = streams[s]->role + "_optical"] {
              // Null once the ring is stopped, which is how this thread exits.
              while (const auto frame = ring->acquire_latest(consumer)) {
                perception::ros_msg::write(
                    *sink, topic,
                    perception::ros_msg::ImageMessage{{frame->timestamp_ns, frame_id},
                                                      {desc, frame->data, frame->bytes}});
              }
            });

        std::printf("recording: %s via a %u-slot host tap (%.1fMB)\n", name.c_str(),
                    streams[s]->tap->slots(),
                    static_cast<double>(streams[s]->tap->slots() * geometry.buffer_bytes) / 1e6);
      }

      perception::ros_msg::Topic<perception::ros_msg::DisparityMessage> disparity_topic;
      // The disparity lives on the rectified LEFT frame, so it shares that
      // eye's optical frame and nothing else's.
      std::string disparity_frame;
      bool have_disparity = false;
      if (config.recording.records(kDisparityTopic) && ess) {
        disparity_frame = streams[0]->role + "_optical";
        perception::ros_msg::DisparityContext disparity_context;
        disparity_context.width = ess->width();
        disparity_context.height = ess->height();
        disparity_context.focal_length_px =
            static_cast<float>(ess->rectification().rectified_fx());
        disparity_context.baseline_m = static_cast<float>(ess->rectification().baseline_m());
        disparity_context.min_disparity = config.ess.display_min_disparity;
        disparity_context.max_disparity = config.ess.display_max_disparity;

        disparity_topic = perception::ros_msg::add_topic<perception::ros_msg::DisparityMessage>(
            *recorder, kDisparityTopic, topic_rate_hz, std::move(disparity_context));
        declared.push_back(kDisparityTopic);
        have_disparity = true;
      }

      // One check for every way a listed topic can have no producer: a typo, an
      // engine that is off, a camera role that does not exist. It names what is
      // available, because the difference is usually one character.
      for (const std::string& wanted : config.recording.topics) {
        if (std::find(declared.begin(), declared.end(), wanted) != declared.end()) continue;
        std::string available;
        for (const std::string& name : declared) available += "\n    " + name;
        if (available.empty()) available = " (nothing)";
        std::printf("FAILED: recording.topics lists '%s', which nothing here produces.%s%s\n",
                    wanted.c_str(),
                    wanted == kDisparityTopic && !ess
                        ? " ess is not enabled here -- `viewer: ess` owns its own engine, so use "
                          "viewer: camera or headless to record disparity."
                        : "",
                    ("\n  Declared:" + available).c_str());
        return 1;
      }

      // After every add_topic(): start() writes no more schema or channel
      // records, and add_topic() after it throws.
      recorder->start();

      if (have_disparity) {
        perception::DownloadStage::Config download_config;
        download_config.slots = config.ess.readback_slots;
        download_config.frame_bytes = ess->pixels() * sizeof(float);
        download_config.width = ess->width();
        download_config.height = ess->height();
        download_config.device_id = config.pipeline.device_id;

        disparity_download = std::make_unique<perception::DownloadStage>(download_config);
        // By value: disparity_topic dies with this block, and a dangling
        // frame_id would be a corrupt field rather than a crash.
        disparity_download->add_sink(
            [&recorder, disparity_topic,
             disparity_frame](const std::shared_ptr<const perception::HostFrame>& frame) {
              // Encodes on the stage's own thread and releases the frame as it
              // returns, so the pinned slot is back before the disk sees it.
              perception::ros_msg::write(
                  *recorder, disparity_topic,
                  perception::ros_msg::DisparityMessage{{frame->timestamp_ns, disparity_frame},
                                                        {frame->data, frame->bytes}});
            });
        disparity_download->start();
        std::printf("recording: disparity %ux%u f32, %u pinned slots (%.1fMB), fx=%.1f "
                    "baseline=%.4fm\n",
                    disparity_topic.context.width, disparity_topic.context.height,
                    config.ess.readback_slots,
                    static_cast<double>(disparity_download->pool().pinned_bytes()) / 1e6,
                    disparity_topic.context.focal_length_px, disparity_topic.context.baseline_m);
      }
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
        stereo->set_pair_callback([&ess, &disparity_download, ref_index](
                                      perception::ReadLease& reference,
                                      perception::ReadLease& other, int64_t skew_ns,
                                      uint64_t pair_id, cudaStream_t stream) {
        perception::ReadLease& left = ref_index == 0 ? reference : other;
        perception::ReadLease& right = ref_index == 0 ? other : reference;

        // Before the drop: the lease is the only thing that knows which frame
        // this disparity is about, and it is about to go away. This stamp is
        // what ties the disparity message to the recorded images.
        const uint64_t reference_timestamp_ns = reference.timestamp_ns();

        ess->preprocess(left.texture(), right.texture(), stream);

        left.drop_hold();
        right.drop_hold();

        ess->infer(stream);

        // Immediately after infer(), on the same stream: that orders the copy
        // behind the inference that produced the disparity and ahead of the
        // next cycle overwriting the binding, which is a single reused buffer
        // with no double buffering. Nothing here blocks -- the copy is async
        // and the stage's own thread does the waiting -- so the pair callback's
        // "do not block in here" contract still holds.
        if (disparity_download) {
          disparity_download->enqueue(ess->disparity_device(), stream, reference_timestamp_ns);
        }

        if (pair_id % 60 != 0) return;

        perception::EssEngine::FrameTiming timing;
        perception::EssEngine::Sample sample;
        const bool have_timing = ess->last_timing(timing);
        const bool have_sample = ess->last_sample(sample);
        std::printf(
            "  ess: pairs=%lu skew=%ldus pre=%.2fms infer=%.2fms total=%.2fms "
            "cycle=%lu d(%u,%u)=%.2fpx z=%.2fm conf=%.2f%s\n",
            static_cast<unsigned long>(pair_id + 1), static_cast<long>(skew_ns / 1000),
            have_timing ? timing.preprocess_ms : -1.0, have_timing ? timing.inference_ms : -1.0,
            have_timing ? timing.total_ms : -1.0,
            static_cast<unsigned long>(sample.index), sample.x, sample.y,
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
      // Order matters and runs upstream-to-downstream: the pair consumer feeds
      // the download stage, which feeds the recorder. Stopping the recorder
      // first would drop the tail of the run; stopping it last means every
      // frame already in flight still lands, and the file is closed on a fully
      // drained queue.
      if (ess_viewer) ess_viewer->stop();
      if (stereo) stereo->stop();
      if (disparity_download) disparity_download->stop();
      // With the sources already stopped, nothing more will be published, so
      // joining the encoders here means every frame that reached the tap is in
      // the queue before close() drains and finalises the file.
      for (auto& stream : streams) stream->stop_tap();
      if (recorder) recorder->close();
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
        stream->source->start(*stream->ring_sink);
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
          if (recorder) {
            std::string line = "  " + recorder->health_line();
            if (disparity_download) line += " | " + disparity_download->health_line();
            for (const auto& stream : streams) {
              if (stream->tap) line += " | " + stream->role + " " + stream->tap->health_line();
            }
            std::printf("%s\n", line.c_str());
          }
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
