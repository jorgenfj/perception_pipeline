#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "acquire_source.hpp"
#include "image_replay_source.hpp"
#include "mcap_player.hpp"

namespace perception {
namespace {

class RecordingAcquireSource final : public AcquireSource {
 public:
  explicit RecordingAcquireSource(const AppConfig& config)
      : config_path_(config.source.path), player_(std::make_unique<McapPlayer>(config.source)) {
    if (player_->image_topics().empty()) {
      throw std::runtime_error("mcap replay: '" + config_path_ +
                               "' holds no sensor_msgs/msg/Image topic to feed the pipeline");
    }

    for (const std::string& topic : player_->image_topics()) {
      sources_.push_back(std::make_unique<ImageReplaySource>(*player_, topic));
    }

    for (std::size_t s = 1; s < sources_.size(); ++s) {
      const CameraGeometry& a = sources_[0]->geometry();
      const CameraGeometry& b = sources_[s]->geometry();
      if (a.width != b.width || a.height != b.height || a.buffer_bytes != b.buffer_bytes) {
        throw std::runtime_error(
            "mcap replay: '" + sources_[0]->topic() + "' is " + std::to_string(a.width) + "x" +
            std::to_string(a.height) + " but '" + sources_[s]->topic() + "' is " +
            std::to_string(b.width) + "x" + std::to_string(b.height) +
            "; one recording cannot feed the pipeline two geometries");
      }
    }
  }

  uint32_t stream_count() const override { return static_cast<uint32_t>(sources_.size()); }

  FrameSource& source(uint32_t stream) override { return *sources_.at(stream); }

  std::string describe(uint32_t stream) const override {
    const ImageReplaySource& s = *sources_.at(stream);
    std::string what = s.topic();
    if (s.message_count() > 0) what += ", " + std::to_string(s.message_count()) + " frames";
    if (!s.frame_id().empty()) what += ", frame " + s.frame_id();
    return "mcap " + config_path_ + " (" + what + ")";
  }

  std::string topic_name(uint32_t stream) const override { return sources_.at(stream)->topic(); }

  std::string frame_id(uint32_t stream) const override { return sources_.at(stream)->frame_id(); }

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
  std::string trigger_health_line() const override { return player_->health_line(); }

 private:
  std::string config_path_;
  std::unique_ptr<McapPlayer> player_;
  std::vector<std::unique_ptr<ImageReplaySource>> sources_;
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
