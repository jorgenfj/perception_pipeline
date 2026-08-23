#include <Spinnaker.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "acquire_source.hpp"
#include "action_sync.hpp"
#include "spinnaker_source.hpp"

namespace perception {
namespace {

class SpinnakerAcquireSource final : public AcquireSource {
 public:
  explicit SpinnakerAcquireSource(const AppConfig& config)
      : system_(Spinnaker::System::GetInstance()), cameras_(system_->GetCameras()) {
    if (config.streams.empty()) throw std::runtime_error("no streams configured");

    std::vector<Spinnaker::CameraPtr> selected;
    for (const StreamConfig& stream : config.streams) {
      selected.push_back(SpinnakerSource::select(cameras_, stream.serial));
    }
    for (std::size_t a = 0; a < selected.size(); ++a) {
      for (std::size_t b = a + 1; b < selected.size(); ++b) {
        if (&*selected[a] != &*selected[b]) continue;
        throw std::runtime_error(
            "streams[" + std::to_string(a) + "] (" + config.streams[a].role + ") and streams[" +
            std::to_string(b) + "] (" + config.streams[b].role +
            ") both resolved to the same camera -- set both serials in `streams:`, or attach "
            "the second camera");
      }
    }

    for (std::size_t s = 0; s < config.streams.size(); ++s) {
      CameraConfig camera = config.camera;
      camera.serial = config.streams[s].serial;
      streams_.push_back(Stream{config.streams[s].role, config.streams[s].serial,
                                std::make_unique<SpinnakerSource>(selected[s], camera)});
    }

    for (std::size_t s = 1; s < streams_.size(); ++s) {
      const CameraGeometry& a = streams_[0].source->geometry();
      const CameraGeometry& b = streams_[s].source->geometry();
      if (a.width != b.width || a.height != b.height || a.pixel_format != b.pixel_format) {
        throw std::runtime_error(
            "stream 0 (" + streams_[0].role + ") came up " + std::to_string(a.width) + "x" +
            std::to_string(a.height) + " " + a.pixel_format + " but stream " +
            std::to_string(s) + " (" + streams_[s].role + ") came up " +
            std::to_string(b.width) + "x" + std::to_string(b.height) + " " + b.pixel_format +
            " -- the geometry in camera.features applies to both, so this means one camera "
            "rejected part of it");
      }
    }
  }

  ~SpinnakerAcquireSource() override {
    streams_.clear();
    cameras_.Clear();
    system_->ReleaseInstance();
  }

  uint32_t stream_count() const override { return static_cast<uint32_t>(streams_.size()); }

  FrameSource& source(uint32_t stream) override { return *streams_.at(stream).source; }

  std::string describe(uint32_t stream) const override {
    const Stream& s = streams_.at(stream);
    const std::string what =
        s.serial.empty() ? std::string("camera (first found)") : "camera " + s.serial;
    return s.role + ": " + what;
  }

  void arm_action_sync(const ActionSyncConfig& config,
                       const std::vector<ActionSyncChecker*>& checkers) override {
    if (!config.enabled) return;

    std::vector<std::string> ptp;
    for (const Stream& s : streams_) ptp.push_back(s.source->ptp_status());

    perception::arm_action_sync(system_, ptp, config, *checkers.at(0));
    checkers.at(0)->label = streams_[0].role;

    for (std::size_t s = 1; s < checkers.size() && s < streams_.size(); ++s) {
      ActionSyncChecker& other = *checkers[s];
      other.enabled = checkers[0]->enabled;
      other.target_ns = checkers[0]->target_ns;
      other.period_ns = checkers[0]->period_ns;
      other.tolerance_ms = checkers[0]->tolerance_ms;
      other.check_frames = checkers[0]->check_frames;
      other.expected_start_offset_ms = checkers[0]->expected_start_offset_ms;
      other.label = streams_[s].role;
    }
  }

 private:
  struct Stream {
    std::string role;
    std::string serial;
    std::unique_ptr<SpinnakerSource> source;
  };

  Spinnaker::SystemPtr system_;
  Spinnaker::CameraList cameras_;
  std::vector<Stream> streams_;
};

}  // namespace

std::unique_ptr<AcquireSource> make_acquire_source(const AppConfig& config) {
  return std::make_unique<SpinnakerAcquireSource>(config);
}

const char* acquire_source_kind() { return "spinnaker"; }

}  // namespace perception
