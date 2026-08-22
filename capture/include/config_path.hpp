#pragma once

#include <string>

namespace perception {

// Prefer a config/ directory next to the binary -- an installed tree ships one
// -- and fall back to `source_config_dir`, which each tool passes as its own
// PERCEPTION_CONFIG_DIR.
//
// Not a build-time copy into bin/: every binary in this project shares one
// bin/, so three tools copying their own config there would fight over one
// directory. Compiling the source path in also means editing a config takes
// effect without a rebuild.
std::string resolve_config_path(const std::string& name, const std::string& source_config_dir);

}  // namespace perception
