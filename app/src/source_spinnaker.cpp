// acquire's frame source when built with -DPERCEPTION_SOURCE=spinnaker: a live
// camera. The only translation unit in app/ that includes <Spinnaker.h>.

#include <Spinnaker.h>

#include <memory>
#include <string>

#include "acquire_source.hpp"
#include "action_sync.hpp"
#include "spinnaker_source.hpp"

namespace perception {
namespace {

class SpinnakerAcquireSource final : public AcquireSource {
 public:
  explicit SpinnakerAcquireSource(const AppConfig& config)
      : system_(Spinnaker::System::GetInstance()),
        cameras_(system_->GetCameras()),
        serial_(config.camera.serial),
        source_(std::make_unique<SpinnakerSource>(
            SpinnakerSource::select(cameras_, config.camera.serial), config.camera)) {}

  // The source goes first -- destroying it stops acquisition and drops the
  // camera handle -- and only then may the list be cleared and the system
  // released. Explicit in the body rather than by member order, because the
  // body runs *before* members are destroyed.
  ~SpinnakerAcquireSource() override {
    source_.reset();
    cameras_.Clear();
    system_->ReleaseInstance();
  }

  FrameSource& source() override { return *source_; }

  std::string describe() const override {
    return serial_.empty() ? std::string("camera (first found)") : "camera " + serial_;
  }

  void arm_action_sync(const ActionSyncConfig& config, ActionSyncChecker& checker) override {
    if (!config.enabled) return;
    // Qualified: the member above shares the name. This is the free function
    // in spinnaker/action_sync.hpp that sends the broadcast.
    perception::arm_action_sync(system_, {source_->ptp_status()}, config, checker);
  }

 private:
  Spinnaker::SystemPtr system_;
  Spinnaker::CameraList cameras_;
  std::string serial_;
  std::unique_ptr<SpinnakerSource> source_;
};

}  // namespace

std::unique_ptr<AcquireSource> make_acquire_source(const AppConfig& config) {
  return std::make_unique<SpinnakerAcquireSource>(config);
}

const char* acquire_source_kind() { return "spinnaker"; }

}  // namespace perception
