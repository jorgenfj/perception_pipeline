// Standalone camera driver: config -> camera -> heap buffers, no CUDA and no
// pipeline anywhere in the link.
//
//   spin_acquire [options] [config.yaml]
//
// This is the bring-up tool. It answers, on the target hardware and without a
// GPU in the picture, the questions that decide whether the zero-copy path is
// viable at all: does this camera accept user-owned buffers with the configured
// stream mode, does it actually fill them, and does the config apply cleanly.
//
// With --record it also writes what the camera delivered, in the format
// recording/ reads -- so a single camera can be captured on a box with no GPU
// and replayed through the whole pipeline later (-DPERCEPTION_SOURCE=recording).

#include <Spinnaker.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "camera_config.hpp"
#include "config_path.hpp"
#include "heap_frame_sink.hpp"
#include "recording_writer.hpp"
#include "spinnaker_source.hpp"

namespace {

volatile std::sig_atomic_t g_stop = 0;
void onSignal(int) { g_stop = 1; }

// This tool's own config directory, the fallback when there is no config/ next
// to the binary. See capture/include/config_path.hpp.
#ifndef PERCEPTION_CONFIG_DIR
#define PERCEPTION_CONFIG_DIR ""
#endif

std::string config_file(const char* name) {
  return perception::resolve_config_path(name, PERCEPTION_CONFIG_DIR);
}

struct Options {
  std::string config_path;
  bool record = false;
  uint64_t max_frames = 0;  // 0 means "whatever the config says"
};

void print_usage() {
  std::printf(
      "spin_acquire [options] [config.yaml]\n"
      "\n"
      "  --record      write every delivered frame to a recording\n"
      "  --frames N    stop after N frames, overriding standalone.max_frames\n");
}

// Unknown arguments are reported rather than ignored: a silently dropped
// --record is a run you have to do twice.
Options parse_args(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage();
      std::exit(0);
    } else if (arg == "--record") {
      options.record = true;
    } else if (arg == "--frames") {
      if (i + 1 >= argc) throw std::runtime_error("--frames needs a value");
      options.max_frames = std::stoull(argv[++i]);
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option '" + arg + "'");
    } else if (options.config_path.empty()) {
      options.config_path = arg;
    } else {
      throw std::runtime_error("more than one config file given ('" + arg + "')");
    }
  }
  return options;
}

// The single stream this tool records, described for the manifest. Everything
// else -- the id, the file names, the padded stride -- RecordingWriter fills in.
perception::StreamInfo stream_info(const perception::CameraGeometry& geometry,
                                   const perception::StandaloneConfig& standalone,
                                   const std::string& serial) {
  perception::StreamInfo info;
  info.role = standalone.record_role;
  info.serial = serial;
  info.width = geometry.width;
  info.height = geometry.height;
  info.stride_bytes = geometry.stride_bytes;
  info.frame_bytes = geometry.frame_bytes;
  info.pixel_format = geometry.pixel_format;
  return info;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, onSignal);

  int status = 0;
  try {
    const Options options = parse_args(argc, argv);
    const std::string config_path =
        options.config_path.empty() ? config_file("camera.yaml") : options.config_path;

    Spinnaker::SystemPtr system = Spinnaker::System::GetInstance();
    Spinnaker::CameraList cameras = system->GetCameras();

    struct SpinnakerGuard {
      Spinnaker::SystemPtr& system;
      Spinnaker::CameraList& cameras;
      ~SpinnakerGuard() {
        cameras.Clear();
        system->ReleaseInstance();
      }
    } spinnaker_guard{system, cameras};

    const perception::CameraConfig camera_config = perception::load_camera_config(config_path);
    perception::StandaloneConfig standalone = perception::load_standalone_config(config_path);
    if (options.record) standalone.record = true;
    if (options.max_frames != 0) standalone.max_frames = options.max_frames;
    std::printf("config: %s\n", config_path.c_str());

    perception::SpinnakerSource source(
        perception::SpinnakerSource::select(cameras, camera_config.serial), camera_config);

    const perception::CameraGeometry& geometry = source.geometry();
    std::printf("camera: %ux%u stride=%u %s, %zu bytes/frame, %zu bytes/buffer, min %u buffers\n",
                geometry.width, geometry.height, geometry.stride_bytes,
                geometry.pixel_format.c_str(), geometry.frame_bytes, geometry.buffer_bytes,
                source.min_slot_count());

    if (standalone.buffer_count < source.min_slot_count()) {
      throw std::runtime_error("standalone.buffer_count is below what this stream mode needs");
    }

    // Opened before acquisition starts, so the very first frame has somewhere
    // to go. The manifest is written at close(), because it carries the frame
    // count and the epoch, neither of which is known until then.
    std::unique_ptr<perception::RecordingWriter> writer;
    if (standalone.record) {
      perception::RecordingWriter::Config writer_config;
      writer_config.root = standalone.record_root;
      writer_config.staging_frames = standalone.staging_frames;

      writer = std::make_unique<perception::RecordingWriter>(
          writer_config,
          std::vector<perception::StreamInfo>{
              stream_info(geometry, standalone, camera_config.serial)});
      writer->set_camera_features(camera_config.features);
      writer->set_ptp_status(source.ptp_status());
      std::printf("recording -> %s\n", writer->directory().c_str());
    }

    perception::HeapFrameSink sink(standalone.buffer_count, geometry.buffer_bytes);
    source.start(sink);

    uint64_t consumed = 0;
    try {
      while (!g_stop && (standalone.max_frames == 0 || consumed < standalone.max_frames)) {
        perception::HeapFrameSink::Frame frame;
        if (!sink.pop(frame, std::chrono::milliseconds(100))) continue;

        // A real reader would work on frame.data here. Without a recorder the
        // slot goes straight back, so the camera never waits on us and the
        // measurement stays isolated to the transport; push() adds one memcpy
        // into the writer's staging ring and never touches the disk on this
        // thread, which is the smallest hold recording can cost.
        if (writer) {
          writer->push(0, frame.meta.timestamp_ns, frame.meta.host_recv_ns, frame.meta.frame_id,
                       frame.data, frame.meta.bytes);
        }
        sink.release(frame.slot);
        ++consumed;

        if (consumed % 60 == 0) {
          std::printf("t=%luns slot=%u bytes=%zu delivered=%lu consumed=%lu incomplete=%lu "
                      "foreign=%lu timeouts=%lu",
                      frame.meta.timestamp_ns, frame.slot, frame.meta.bytes, source.delivered(),
                      consumed,
                      source.incomplete(), source.foreign(), source.timeouts());
          if (writer) std::printf(" | %s", writer->health_line().c_str());
          std::printf("\n");
        }
      }
    } catch (...) {
      source.stop();
      sink.stop();
      throw;
    }

    source.stop();
    sink.stop();

    // After the camera has stopped, so nothing is still being pushed while the
    // staging rings drain.
    if (writer) {
      writer->close();
      std::printf("\nrecording: %s (%s)\n", writer->directory().c_str(),
                  writer->health_line().c_str());
    }

    std::printf("\ndelivered=%lu consumed=%lu incomplete=%lu foreign=%lu timeouts=%lu\n",
                source.delivered(), consumed, source.incomplete(), source.foreign(),
                source.timeouts());

    // The whole point of the tool: a frame that arrived in a buffer we
    // allocated proves the transport honoured SetUserBuffers.
    if (source.delivered() == 0) {
      std::printf("\nNO FRAMES: check BeginAcquisition errors above. "
                  "SPINNAKER_ERR_NOT_IMPLEMENTED means this stream mode rejects user buffers.\n");
      status = 1;
    } else if (source.foreign() > 0) {
      std::printf("\nZERO-COPY NOT IN EFFECT: %lu frames arrived in the library's own buffers.\n",
                  source.foreign());
      status = 1;
    } else {
      std::printf("\nzero-copy confirmed: every frame landed in a user buffer\n");
    }

    // A recording that dropped frames is still a valid recording -- the gaps are
    // visible in the index -- but it is not the clean capture that was asked
    // for, and saying so beats finding out at playback.
    if (writer && writer->stream(0).drops() > 0) {
      std::printf("\nRECORDING INCOMPLETE: %lu frames were dropped before reaching the disk. "
                  "The disk could not keep up; see staging_peak and write_max above.\n",
                  writer->stream(0).drops());
      status = 1;
    }
    if (writer && !writer->stream(0).error().empty()) {
      std::printf("\nRECORDING FAILED: %s\n", writer->stream(0).error().c_str());
      status = 1;
    }
  } catch (const std::exception& e) {
    std::printf("FAILED: %s\n", e.what());
    status = 1;
  }

  return status;
}
