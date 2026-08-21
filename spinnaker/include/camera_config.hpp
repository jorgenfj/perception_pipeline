#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace perception {

// A GenICam node and the value to write to it, kept as text because that is
// what a config file has. The node's own type decides how the text is parsed,
// so one list covers integers, floats, enums, booleans and commands.
using FeatureList = std::vector<std::pair<std::string, std::string>>;

struct CameraConfig {
  std::string serial;          // empty picks the first camera found
  uint64_t timeout_ms = 1000;  // GetNextImage timeout

  // Applied in order, before acquisition starts. Order is significant on real
  // cameras: ExposureAuto has to be Off before ExposureTime is writable, and
  // geometry has to be set before PayloadSize is meaningful.
  FeatureList features;

  // The same, against the transport-layer stream node map.
  FeatureList stream_features;

  // A camera error ends acquisition, not the run: the source tears the stream
  // down and re-opens it. Negative retries forever, 0 disables reconnect (the
  // first non-timeout error is then fatal, as it used to be).
  //
  // "Attempts" counts consecutive failures to get back to acquiring
  int reconnect_attempts = -1;

  // Delay before the first retry, doubled each consecutive failure up to
  // reconnect_backoff_max_ms. The first retry after a good run is immediate.
  uint64_t reconnect_backoff_ms = 500;
  uint64_t reconnect_backoff_max_ms = 5000;
};

// What the camera turned out to be once the features were applied. The pixel
// format stays a GenICam symbolic name rather than a pipeline enum -- mapping
// vendor names onto pipeline types is the composing application's job, and
// keeping it out of here is what leaves this subproject standalone.
struct CameraGeometry {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t stride_bytes = 0;
  std::string pixel_format;

  // PayloadSize: the exact per-frame transfer size, padding included.
  std::size_t frame_bytes = 0;

  // frame_bytes rounded up to the USB3 packet size, which is what a sink slot
  // must be. Sizing below this tears images on USB3 cameras.
  std::size_t buffer_bytes = 0;
};

// Reads the `camera:` section of a YAML document, ignoring everything else, so
// the same file can carry an application's own sections alongside it. Throws
// std::runtime_error naming the offending key rather than falling back to a
// default. A document with no `camera:` section yields the defaults above.
CameraConfig load_camera_config(const std::string& path);

// How many buffers the standalone driver allocates when the config does not
// say. Only used by spin_acquire; the pipeline sizes its own rings.
struct StandaloneConfig {
  uint32_t buffer_count = 8;
  uint64_t max_frames = 0;  // 0 runs until interrupted
};

StandaloneConfig load_standalone_config(const std::string& path);

}  // namespace perception
