#include "config_path.hpp"

#include <filesystem>

namespace perception {

std::string resolve_config_path(const std::string& name, const std::string& source_config_dir) {
  std::error_code ec;
  const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (!ec) {
    const std::filesystem::path beside = exe.parent_path() / "config" / name;
    if (std::filesystem::exists(beside, ec)) return beside.string();
  }
  if (!source_config_dir.empty()) {
    return (std::filesystem::path(source_config_dir) / name).string();
  }
  return (std::filesystem::current_path() / "config" / name).string();
}

}  // namespace perception
