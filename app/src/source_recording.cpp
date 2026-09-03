#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "acquire_source.hpp"
#include "mcap_replay_source.hpp"

namespace perception {
namespace {

class RecordingAcquireSource final : public AcquireSource {
 public:
  explicit RecordingAcquireSource(const AppConfig& config)
      : config_path_(config.source.path),
        source_(std::make_unique<McapReplaySource>(config.source)) {}

  // Always one. McapReplaySource replays a single topic by design.
  uint32_t stream_count() const override { return 1; }

  FrameSource& source(uint32_t) override { return *source_; }

  std::string describe(uint32_t) const override {
    std::string what = source_->topic();
    if (source_->message_count() > 0) {
      what += ", " + std::to_string(source_->message_count()) + " frames";
    }
    if (!source_->frame_id().empty()) what += ", frame " + source_->frame_id();
    return "mcap " + config_path_ + " (" + what + ")";
  }

  int64_t epoch_offset_ns() const override { return 0; }

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

  void stop_action_sync() override {}
  std::string trigger_health_line() const override { return {}; }

 private:
  std::string config_path_;
  std::unique_ptr<McapReplaySource> source_;
};

}  // namespace

std::unique_ptr<AcquireSource> make_acquire_source(const AppConfig& config) {
  if (config.source.path.empty()) {
    throw std::runtime_error(
        "this build replays recordings (-DPERCEPTION_SOURCE=recording) but no recording was "
        "given: set source.recording in the config to an .mcap file");
  }
  return std::make_unique<RecordingAcquireSource>(config);
}

const char* acquire_source_kind() { return "recording"; }

}  // namespace perception
