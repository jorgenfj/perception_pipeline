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
  uint64_t max_frames = 0;  // 0 means "whatever the config says"
};

void print_usage() {
  std::printf(
      "spin_acquire [options] [config.yaml]\n"
      "\n"
      "  --frames N    stop after N frames, overriding standalone.max_frames\n");
}

Options parse_args(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage();
      std::exit(0);
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

    perception::HeapFrameSink sink(standalone.buffer_count, geometry.buffer_bytes);
    source.start(sink);

    uint64_t consumed = 0;
    try {
      while (!g_stop && (standalone.max_frames == 0 || consumed < standalone.max_frames)) {
        perception::HeapFrameSink::Frame frame;
        if (!sink.pop(frame, std::chrono::milliseconds(100))) continue;

        sink.release(frame.slot);
        ++consumed;

        if (consumed % 60 == 0) {
          std::printf("t=%luns slot=%u bytes=%zu delivered=%lu consumed=%lu incomplete=%lu "
                      "foreign=%lu timeouts=%lu",
                      frame.meta.timestamp_ns, frame.slot, frame.meta.bytes, source.delivered(),
                      consumed,
                      source.incomplete(), source.foreign(), source.timeouts());
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
  } catch (const std::exception& e) {
    std::printf("FAILED: %s\n", e.what());
    status = 1;
  }

  return status;
}
