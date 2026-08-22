// acquire's frame source when built with -DPERCEPTION_SOURCE=recording: one
// stream of a recording, replayed at the pacing it was captured with. No camera
// and no <Spinnaker.h> in this translation unit.

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#include "acquire_source.hpp"
#include "recording_source.hpp"

namespace perception {
namespace {

class RecordingAcquireSource final : public AcquireSource {
 public:
  explicit RecordingAcquireSource(const AppConfig& config)
      : source_(std::make_unique<RecordingSource>(config.source)) {}

  FrameSource& source() override { return *source_; }

  std::string describe() const override {
    const StreamInfo& info = source_->reader().stream(source_->stream());
    return "recording " + source_->reader().manifest().created_utc + " stream " +
           std::to_string(source_->stream()) + " (" + info.role + ", " + info.serial + ")";
  }

  void arm_action_sync(const ActionSyncConfig& config, ActionSyncChecker& checker) override {
    if (!config.enabled) return;
    // Not an error: action_sync is a rig setting and one config file serves
    // both builds. Left disabled so the report line does not claim to be
    // checking a trigger that never fired.
    checker.enabled = false;
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
