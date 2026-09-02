#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "mcap_recorder.hpp"

namespace perception {

/**
 * @brief The `recording:` section: whether to record, where, and what.
 */
struct RecordingConfig {
  bool enabled = false;

  std::vector<std::string> topics;

  McapRecorder::Config recorder;

  /** @brief True if `topic` is listed. */
  bool records(std::string_view topic) const;
};

/**
 * @brief Read the `recording:` section, ignoring every other section.
 * @param path YAML document.
 * @return The parsed section, or defaults if it is absent.
 *
 * @throws std::runtime_error naming the offending key. Whether a listed topic
 *         has a producer cannot be known here; the app checks that once it has
 *         declared them.
 */
RecordingConfig load_recording_config(const std::string& path);

}  // namespace perception
