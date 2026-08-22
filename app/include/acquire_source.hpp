#pragma once

#include <memory>
#include <string>

#include "action_sync_check.hpp"
#include "app_config.hpp"
#include "frame_source.hpp"

namespace perception {

// The frame source `acquire` was built with, plus whatever runtime has to stay
// alive around it. Exactly one of app/src/source_spinnaker.cpp and
// app/src/source_recording.cpp is compiled in, chosen by -DPERCEPTION_SOURCE.
//
// It exists rather than acquire_main.cpp constructing a SpinnakerSource
// directly because the vendor build has a SystemPtr and a CameraList whose
// lifetime must bracket the source, and the recording build must not mention
// either. One owner for both keeps acquire_main.cpp free of <Spinnaker.h>.
class AcquireSource {
 public:
  virtual ~AcquireSource() = default;

  // Ready to start() by the time the factory returns, so geometry() can size
  // the ingress ring.
  virtual FrameSource& source() = 0;

  // What this source turned out to be, for the startup line: a camera serial,
  // or a recording directory and which stream of it.
  virtual std::string describe() const = 0;

  // A scheduled AcquisitionStart, when the config asks for one. Called after
  // start(): with TriggerMode=On/AcquisitionStart the camera delivers nothing
  // until the command fires, so BeginAcquisition has to have run first.
  //
  // A source that cannot schedule anything says so and carries on rather than
  // throwing -- action_sync left enabled while replaying a recording is a
  // config written for the rig, not an error.
  virtual void arm_action_sync(const ActionSyncConfig& config, ActionSyncChecker& checker) = 0;
};

// Defined by whichever source_*.cpp was compiled in.
std::unique_ptr<AcquireSource> make_acquire_source(const AppConfig& config);

// "spinnaker" or "recording". A build-time fact, and a run whose log does not
// record which it was is a log you cannot read six months later.
const char* acquire_source_kind();

}  // namespace perception
