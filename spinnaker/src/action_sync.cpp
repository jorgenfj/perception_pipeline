#include "action_sync.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "frame_sink.hpp"

namespace perception {
namespace {

const char* action_command_status_name(Spinnaker::ActionCommandStatus status) {
  switch (status) {
    case Spinnaker::SPINNAKER_ACTION_COMMAND_STATUS_OK:
      return "OK";
    case Spinnaker::SPINNAKER_ACTION_COMMAND_STATUS_NO_REF_TIME:
      return "NO_REF_TIME (camera is not PTP-locked)";
    case Spinnaker::SPINNAKER_ACTION_COMMAND_STATUS_OVERFLOW:
      return "OVERFLOW (scheduled-action queue full)";
    case Spinnaker::SPINNAKER_ACTION_COMMAND_STATUS_ACTION_LATE:
      return "ACTION_LATE (scheduled time was already in the past)";
    case Spinnaker::SPINNAKER_ACTION_COMMAND_STATUS_ERROR:
      return "ERROR";
    default:
      return "unknown";
  }
}

}  // namespace

void arm_action_sync(Spinnaker::SystemPtr& system, const std::vector<std::string>& ptp_statuses,
                     const ActionSyncConfig& cfg, ActionSyncChecker& checker) {
  // TriggerMode=On/AcquisitionStart means the camera delivers nothing at all
  // until this fires -- BeginAcquisition() (already run by the caller) just
  // arms the stream and returns; GetNextImage() times out quietly until the
  // scheduled instant.
  for (std::size_t i = 0; i < ptp_statuses.size(); ++i) {
    if (ptp_statuses[i] == "Slave") continue;
    std::string joined;
    for (const std::string& status : ptp_statuses) {
      if (!joined.empty()) joined += "/";
      joined += status.empty() ? "unsupported" : status;
    }
    throw std::runtime_error(
        "action_sync: camera " + std::to_string(i) + " PTP status is '" + ptp_statuses[i] +
        "' (rig reads " + joined +
        "), not Slave -- a scheduled Action Command on an unsynced camera comes back "
        "NO_REF_TIME. 'Master' means the camera elected itself because it heard no "
        "grandmaster: bring up ptp4l on the host first (see spinnaker/README.md).");
  }

  if (!ptp_timebase_ready()) {
    std::printf("action_sync: warning -- the kernel holds no TAI offset, so the scheduled "
                "instant is being built in UTC and the cameras will read it ~37s late. Is "
                "phc2sys running?\n");
  }
  const uint64_t now_ns = ptp_now_ns();
  const uint64_t earliest_ns = now_ns + static_cast<uint64_t>(cfg.lead_time_ms * 1e6);
  const uint64_t target_ns = ((earliest_ns / 1'000'000'000ull) + 1) * 1'000'000'000ull;

  Spinnaker::ActionCommandResult results[8];
  unsigned int result_size = 8;
  system->SendActionCommand(cfg.device_key, cfg.group_key, cfg.group_mask, target_ns,
                            /*requestAck=*/true, &result_size, results);

  std::printf("action_sync: scheduled AcquisitionStart for t=%luns TAI (%.0fms from now, "
              "TAI-UTC=%lds)\n", target_ns, static_cast<double>(target_ns - now_ns) * 1e-6,
              static_cast<long>(tai_offset_s()));
  for (unsigned int i = 0; i < result_size; ++i) {
    std::printf("action_sync: ack from %u.%u.%u.%u: %s\n",
                (results[i].DeviceAddress >> 24) & 0xFFu, (results[i].DeviceAddress >> 16) & 0xFFu,
                (results[i].DeviceAddress >> 8) & 0xFFu, results[i].DeviceAddress & 0xFFu,
                action_command_status_name(results[i].Status));
  }
  if (result_size == 0) {
    std::printf("action_sync: no acks received -- camera may not support Action Commands, or "
                "ActionDeviceKey/GroupKey/GroupMask don't match the config\n");
  }

  checker.enabled = true;
  checker.target_ns = target_ns;
  checker.period_ns = 1e9 / cfg.expected_hz;
  checker.tolerance_ms = cfg.tolerance_ms;
  checker.check_frames = cfg.check_frames;
  checker.expected_start_offset_ms = cfg.expected_start_offset_ms;
}

}  // namespace perception
