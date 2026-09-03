#pragma once

#include <string>

#include "mcap_player.hpp"

namespace perception {

/**
 * @brief Read the `source:` section, ignoring every other section.
 *
 * @param path YAML document.
 * @return The parsed section, or defaults if it is absent.
 *
 * @throws std::runtime_error naming the offending key. Whether a listed topic
 *         is in the file cannot be known here; McapPlayer checks that when it
 *         opens one.
 */
McapPlayer::Config load_replay_config(const std::string& path);

}  // namespace perception
