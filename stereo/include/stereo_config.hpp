#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace perception {

// The `stereo:`, `recording:` and `streams:` sections of a config file. The
// `camera:` section alongside them is read by load_camera_config() and applies
// to both cameras -- exposure, gain, geometry and PTP are rig-wide settings,
// and two cameras configured differently would not be a stereo pair. Only the
// serial differs per stream, which is why that is all `streams:` carries.

struct StereoStreamConfig {
  // "left" / "right". A label carried into the manifest for humans; nothing
  // branches on it, and the pairing does not know which is which.
  std::string role;

  // Empty picks whatever camera is found first, which is only useful with a
  // single camera attached -- for a real rig both serials must be set, or the
  // two streams may swap between runs.
  std::string serial;
};

struct StereoConfig {
  // Pairing. tolerance must stay strictly under half the frame period; see
  // require_pair_tolerance().
  uint64_t tolerance_ns = 500'000;
  uint64_t frame_period_ns = 0;  // 0 skips the tolerance check
  uint32_t queue_frames = 8;
  uint32_t hold_ms = 40;

  // Display. `decimate` is the Bayer block that becomes one displayed pixel;
  // see debayer_to_rgb() for why 1 is not on offer.
  bool display = true;
  uint32_t decimate = 2;
  uint32_t window_width = 1600;
  uint32_t window_height = 700;
  bool vsync = true;

  // Camera buffers per stream. Must be at least what the stream mode needs.
  uint32_t buffer_count = 8;

  // Recording. Off by default: this is a viewer that can record, not a
  // recorder with a preview.
  bool record = false;
  std::string record_root = "recordings";
  uint32_t staging_frames = 32;

  std::vector<StereoStreamConfig> streams;
};

// Reads the sections above, ignoring everything else, so one file can carry the
// camera schema and this alongside each other. Throws naming the offending key
// rather than falling back to a default.
StereoConfig load_stereo_config(const std::string& path);

}  // namespace perception
