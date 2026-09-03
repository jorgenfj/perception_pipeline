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
#include "topic_graph.hpp"
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

constexpr const char* kDisparityTopic = "/disparity";

// "/left/image_raw" -> "/left/image_color". The device ring holds the debayered
// form of whatever produced the raw frames, so it is named under the same
// namespace rather than rebuilt from the role: a replayed file may carry a
// topic the config never mentions, and the two names have to stay together.
std::string sibling_topic(const std::string& topic, const char* leaf) {
  const std::size_t slash = topic.rfind('/');
  const std::string space =
      (slash == 0 || slash == std::string::npos) ? topic : topic.substr(0, slash);
  return space + "/" + leaf;
}

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
          "        A replay build opens one stream per image topic it was pointed at; see\n"
          "        source.image_topics in the config.\n",
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

    // Every buffer this run produces, under the name it produces it as. Declared
    // before `streams` so it outlives nothing it points at, and resolved only
    // during setup -- nothing below looks a string up on a per-frame path.
    perception::TopicGraph graph;

    // The device ring each stream's debayered frames land in, indexed the same
    // way `streams` is, so the pairing can name the partner of its reference.
    std::vector<std::string> colour_topics;

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

      const std::string frame_id = acquire_source->frame_id(s);
      streams.push_back(std::move(stream));

      const std::string raw_topic = acquire_source->topic_name(s);

      perception::TopicInfo raw;
      raw.name = raw_topic;
      raw.residency = perception::Residency::Host;
      raw.ros_type = std::string(perception::ros_msg::schema::kImageType);
      raw.frame_id = frame_id;
      raw.producer = acquire_source->describe(s);
      // The tap is the CPU fan-out, and it exists only if something asks for
      // it: with no host-side consumer there is no ring and no copy.
      graph.declare_image_stream(
          std::move(raw), s, [&streams, s, geometry, desc]() -> perception::HostFrameRing& {
            StreamPipeline& bound = *streams[s];
            bound.tap = std::make_unique<perception::HostFrameRing>(
                kTapConsumers + 2, geometry.buffer_bytes, desc.width, desc.height);
            bound.ring_sink->set_tap(bound.tap.get());
            return *bound.tap;
          });

      perception::TopicInfo colour;
      colour.name = sibling_topic(raw_topic, "image_color");
      colour.frame_id = frame_id;
      colour.producer = debayer.name();
      colour_topics.push_back(colour.name);
      graph.declare_device_ring(std::move(colour), s,
                                [&streams, s]() -> perception::DeviceRingBuffer& {
                                  return *streams[s]->device_ring;
                                });
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

    // Resolved by name once, held as a reference from here on: that is the
    // whole contract of the graph, and it is why naming the dataflow costs
    // nothing on a per-frame path.
    perception::DeviceRingBuffer& reference_ring = graph.device_ring(colour_topics[0]);

    perception::YoloConfig yolo_config = config.yolo;
    yolo_config.engine_path = perception::resolve_next_to_exe(yolo_config.engine_path);

    std::unique_ptr<perception::ViewerConsumer> camera_viewer;
    std::unique_ptr<perception::YoloViewerConsumer> yolo_viewer;
    std::unique_ptr<perception::HeadlessYoloConsumer> headless_yolo;

    switch (config.viewer_mode) {
      case perception::ViewerMode::Camera:
        camera_viewer = std::make_unique<perception::ViewerConsumer>(
            reference_ring, reference.probe, display_desc, config.display,
            kSecondaryConsumerId, config.pipeline.device_id);
        break;
      case perception::ViewerMode::Yolo:
        yolo_viewer = std::make_unique<perception::YoloViewerConsumer>(
            reference_ring, reference.probe, display_desc, config.display, yolo_config,
            kSecondaryConsumerId, config.pipeline.device_id);
        break;
      case perception::ViewerMode::Headless:
        headless_yolo = std::make_unique<perception::HeadlessYoloConsumer>(
            reference_ring, display_desc, yolo_config,
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

    // Declared whenever there is an engine to produce it, recorded or not. The
    // readback stage behind it is built on first resolve, so a run that nobody
    // subscribes to still allocates no pinned pool.
    perception::ros_msg::DisparityContext disparity_context;
    if (ess) {
      disparity_context.width = ess->width();
      disparity_context.height = ess->height();
      disparity_context.focal_length_px = static_cast<float>(ess->rectification().rectified_fx());
      disparity_context.baseline_m = static_cast<float>(ess->rectification().baseline_m());
      disparity_context.min_disparity = config.ess.display_min_disparity;
      disparity_context.max_disparity = config.ess.display_max_disparity;

      perception::TopicInfo disparity;
      disparity.name = kDisparityTopic;
      disparity.residency = perception::Residency::Host;
      disparity.ros_type = std::string(perception::ros_msg::schema::kDisparityImageType);
      // The disparity lives on the rectified LEFT frame, so it shares that
      // eye's optical frame and nothing else's.
      disparity.frame_id = graph.info(colour_topics[0]).frame_id;
      disparity.producer = "ess";
      graph.declare_host_plane(
          std::move(disparity), [&]() -> perception::DownloadStage& {
            perception::DownloadStage::Config download_config;
            download_config.slots = config.ess.readback_slots;
            download_config.frame_bytes = ess->pixels() * sizeof(float);
            download_config.width = ess->width();
            download_config.height = ess->height();
            download_config.device_id = config.pipeline.device_id;
            disparity_download = std::make_unique<perception::DownloadStage>(download_config);
            return *disparity_download;
          });
    }

    if (recording_enabled) {
      // Nested, so adding a recorder knob touches neither this file nor app_config.
      perception::McapRecorder::Config recorder_config = config.recording.recorder;
      // Not a knob: what the sinks already took off, so the file can say so.
      recorder_config.epoch_offset_ns = acquire_source->epoch_offset_ns();
      recorder = std::make_unique<perception::McapRecorder>(recorder_config);

      // Sizes the queues only, so being a factor out costs a queue depth.
      const double topic_rate_hz = config.action_sync.expected_hz;

      for (const std::string& name : config.recording.topics) {
        // The one case the graph cannot explain on its own: `viewer: ess` owns
        // its own engine, so there is no disparity out here to record.
        if (name == kDisparityTopic && !ess) {
          throw std::runtime_error(
              "recording.topics lists '" + name +
              "', but nothing here produces it -- ess is not enabled, or `viewer: ess` owns the "
              "engine. Use viewer: camera or headless to record disparity.");
        }

        // Every other way a listed topic can have no producer -- a typo, a
        // camera role that does not exist -- comes back from here naming what
        // this run actually declared. The difference is usually one character.
        const perception::TopicInfo& info = graph.info(name);
        if (info.residency != perception::Residency::Host) {
          throw std::runtime_error("recording.topics lists '" + name +
                                   "', which is device-resident. Device buffers have no wire "
                                   "form; record the host topic they were built from.");
        }

        if (info.ros_type == perception::ros_msg::schema::kImageType) {
          // No geometry here: the reservation is the type's own default and each
          // message carries its own desc.
          const auto topic = perception::ros_msg::add_topic<perception::ros_msg::ImageMessage>(
              *recorder, name, topic_rate_hz);

          // Resolving is what builds the tap and wires it into the frame path.
          perception::HostFrameRing& tap = graph.host_tap(name);
          const uint32_t consumer = tap.add_consumer("recorder");

          // Everything the encode needs is captured by value -- the topic, the
          // geometry, the frame name -- because this scope is gone long before
          // the thread is. The frame is held for the whole encode and released
          // as the loop turns, which costs a tap slot and nothing upstream.
          streams.at(graph.stream_index(name))->recorder_thread = std::thread(
              [ring = &tap, consumer, sink = recorder.get(), topic, desc,
               frame_id = info.frame_id] {
                // Null once the ring is stopped, which is how this thread exits.
                while (const auto frame = ring->acquire_latest(consumer)) {
                  perception::ros_msg::write(
                      *sink, topic,
                      perception::ros_msg::ImageMessage{{frame->timestamp_ns, frame_id},
                                                        {desc, frame->data, frame->bytes}});
                }
              });

          std::printf("recording: %s via a %u-slot host tap (%.1fMB)\n", name.c_str(), tap.slots(),
                      static_cast<double>(tap.slots() * geometry.buffer_bytes) / 1e6);
        } else if (info.ros_type == perception::ros_msg::schema::kDisparityImageType) {
          const auto topic = perception::ros_msg::add_topic<perception::ros_msg::DisparityMessage>(
              *recorder, name, topic_rate_hz, disparity_context);

          perception::DownloadStage& download = graph.download(name);
          // By value: this scope dies long before the stage does, and a
          // dangling frame_id would be a corrupt field rather than a crash.
          download.add_sink([sink = recorder.get(), topic, frame_id = info.frame_id](
                                const std::shared_ptr<const perception::HostFrame>& frame) {
            // Encodes on the stage's own thread and releases the frame as it
            // returns, so the pinned slot is back before the disk sees it.
            perception::ros_msg::write(
                *sink, topic,
                perception::ros_msg::DisparityMessage{{frame->timestamp_ns, frame_id},
                                                      {frame->data, frame->bytes}});
          });

          std::printf("recording: %s %ux%u f32, %u pinned slots (%.1fMB), fx=%.1f "
                      "baseline=%.4fm\n",
                      name.c_str(), disparity_context.width, disparity_context.height,
                      config.ess.readback_slots,
                      static_cast<double>(download.pool().pinned_bytes()) / 1e6,
                      disparity_context.focal_length_px, disparity_context.baseline_m);
        } else {
          throw std::runtime_error("recording.topics lists '" + name + "', which carries '" +
                                   info.ros_type + "'. Nothing here knows how to encode that.");
        }
      }

      // After every add_topic(): start() writes no more schema or channel
      // records, and add_topic() after it throws.
      recorder->start();
    }

    // Only if something resolved it above; with no subscriber there is no
    // stage, no pinned pool and no thread.
    if (disparity_download) disparity_download->start();

    std::unique_ptr<perception::RingPairConsumer> stereo;
    // After `stereo`, so its thread -- which this one drives -- is joined
    // before the consumer it is driving goes away.
    std::unique_ptr<perception::EssViewerConsumer> ess_viewer;
    if (stereo_enabled) {
      // Named rather than indexed: which eye anchors the pairing is a property
      // of the topic, so the config says the same thing whether a camera or a
      // recording produced it.
      const uint32_t ref_index = graph.stream_index(config.stereo.reference);
      if (ref_index == perception::TopicGraph::kNoStream) {
        throw std::runtime_error("stereo.reference names '" + config.stereo.reference +
                                 "', which belongs to no stream; it has to be one eye's ring");
      }
      const uint32_t other_index = ref_index == 0 ? 1u : 0u;

      stereo = std::make_unique<perception::RingPairConsumer>(
          graph.device_ring(config.stereo.reference), graph.device_ring(colour_topics[other_index]),
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
          config.stereo.reference.c_str(), colour_topics[other_index].c_str(),
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

    // Every name this run produces, after the last of them is declared. The
    // whole dataflow in one block, and the one place a replay run and a live
    // run can be read side by side.
    std::printf("%s", graph.summary().c_str());

    perception::Reporter reporter(*reference.source, *reference.upload, reference_ring,
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
        const uint64_t seen = reference_ring.wait_seq();

        perception::FramePeek peek;
        if (!reference_ring.view_latest_inplace(peek) ||
            (peek.slot == last_slot && peek.slot_seq == last_seq)) {
          reference_ring.wait_for_publish(seen);
          continue;
        }

        perception::ReadLease lease =
            reference_ring.lease_latest(kPipelineConsumerId, consumer);
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
