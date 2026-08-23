#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "acquire_source.hpp"
#include "recording_source.hpp"

namespace perception {
namespace {

class RecordingAcquireSource final : public AcquireSource {
 public:
  explicit RecordingAcquireSource(const AppConfig& config)
      : source_(std::make_unique<RecordingSource>(config.source)) {}

  // Always one. RecordingSource replays a single stream by design
  uint32_t stream_count() const override { return 1; }

  FrameSource& source(uint32_t) override { return *source_; }

  std::string describe(uint32_t) const override {
    const StreamInfo& info = source_->reader().stream(source_->stream());
    return "recording " + source_->reader().manifest().created_utc + " stream " +
           std::to_string(source_->stream()) + " (" + info.role + ", " + info.serial + ")";
  }

  void arm_action_sync(const ActionSyncConfig& config,
                       const std::vector<ActionSyncChecker*>& checkers) override {
    if (!config.enabled) return;
    // Not an error: action_sync is a rig setting and one config file serves
    // both builds. Left disabled so the report line does not claim to be
    // checking a trigger that never fired.
    for (ActionSyncChecker* checker : checkers) checker->enabled = false;
    std::printf(
        "action_sync: ignored -- this build replays a recording, so there is no camera to\n"
        "             schedule and the frame spacing is the one already in the file\n");
  }

 private:
  std::unique_ptr<RecordingSource> source_;
};

}  // namespace

std::unique_ptr<AcquireSource> make_acquire_source(const AppConfig& config) {
  if (config.source.directory.empty()) {
    throw std::runtime_error(
        "this build replays recordings (-DPERCEPTION_SOURCE=recording) but no recording was "
        "given: set source.recording in the config to a recording directory");
  }
  return std::make_unique<RecordingAcquireSource>(config);
}

const char* acquire_source_kind() { return "recording"; }

}  // namespace perception
