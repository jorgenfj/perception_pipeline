#pragma once

#include <Spinnaker.h>

#include <cstdint>
#include <string>
#include <vector>

#include "action_sync_check.hpp"
#include "camera_config.hpp"

namespace perception {

// Schedules the AcquisitionStart Action Command that `cfg` describes and arms
// `checker` to verify it landed. One broadcast covers every camera matching
// device_key/group_key/group_mask, so a stereo pair takes one call; the
// per-camera acks are printed as they come back.
//
// `ptp_statuses` is every participating camera's GevIEEE1588Status. Throws
// unless all of them read "Slave": a scheduled command on an unsynced camera
// comes back NO_REF_TIME, and on a rig where only one eye is locked it would
// half-work, which is worse than not working.
//
// `checker` is an out-param rather than a return value because the caller owns
// it and keeps feeding it frames long after this returns.
void arm_action_sync(Spinnaker::SystemPtr& system, const std::vector<std::string>& ptp_statuses,
                     const ActionSyncConfig& cfg, ActionSyncChecker& checker);

}  // namespace perception
