#include "action_sync.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "latency_probe.hpp"

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

void ActionSyncChecker::observe(uint64_t ts_ns) {
  if (!enabled || done) return;
  if (n == 0) {
    first_ts = ts_ns;
  } else {
    const double expected_period_ms = period_ns * 1e-6;
    const double actual_gap_ms = static_cast<double>(ts_ns - prev_ts) * 1e-6;
    const double dev_ms = actual_gap_ms - expected_period_ms;
    spacing_min_ms = std::min(spacing_min_ms, dev_ms);
    spacing_max_ms = std::max(spacing_max_ms, dev_ms);
    spacing_sum_ms += dev_ms;
  }
  prev_ts = ts_ns;
  ++n;
  if (n >= check_frames) {
    print_verdict();
    done = true;
  }
}

void ActionSyncChecker::print_verdict() const {
  const double start_offset_ms =
      static_cast<double>(static_cast<int64_t>(first_ts) - static_cast<int64_t>(target_ns)) *
      1e-6;
  const double spacing_mean_ms = n > 1 ? spacing_sum_ms / static_cast<double>(n - 1) : 0.0;
  // Compared against expected_start_offset_ms, not zero: GetTimeStamp()'s
  // latch point relative to the trigger is a sensor convention (e.g.
  // end-of-exposure), not part of what this test is actually checking.
  // Spacing deviation has no such excuse -- it's governed purely by the
  // camera's own free-running timer once started, nothing external.
  const double start_deviation_ms = start_offset_ms - expected_start_offset_ms;
  const bool start_ok = std::fabs(start_deviation_ms) <= tolerance_ms;
  const bool spacing_ok =
      n <= 1 || (std::fabs(spacing_min_ms) <= tolerance_ms && std::fabs(spacing_max_ms) <= tolerance_ms);
  std::printf(
      "\naction_sync: %s -- start offset %.3f ms, expected %.3f ms, deviation %.3f ms "
      "(tolerance %.2f ms); frame spacing deviation from 1/expected_hz min/mean/max "
      "%.3f/%.3f/%.3f ms over %u frames\n",
      (start_ok && spacing_ok) ? "PASS" : "FAIL", start_offset_ms, expected_start_offset_ms,
      start_deviation_ms, tolerance_ms, spacing_min_ms, spacing_mean_ms, spacing_max_ms, n);
}

void arm_action_sync(Spinnaker::SystemPtr& system, SpinnakerSource& source,
                     const ActionSyncConfig& cfg, ActionSyncChecker& checker) {
  // TriggerMode=On/AcquisitionStart means the camera delivers nothing at all
  // until this fires -- BeginAcquisition() (already run by the caller) just
  // arms the stream and returns; GetNextImage() times out quietly until the
  // scheduled instant.
  if (source.ptp_status() != "Slave") {
    throw std::runtime_error(
        "action_sync: camera PTP status is not Slave -- a scheduled Action Command on an "
        "unsynced camera comes back NO_REF_TIME. Bring up ptp4l/phc2sys first (see "
        "spinnaker/README.md).");
  }

  const uint64_t now_ns = LatencyProbe::host_now_ns();
  const uint64_t earliest_ns = now_ns + static_cast<uint64_t>(cfg.lead_time_ms * 1e6);
  const uint64_t target_ns = ((earliest_ns / 1'000'000'000ull) + 1) * 1'000'000'000ull;

  Spinnaker::ActionCommandResult results[8];
  unsigned int result_size = 8;
  system->SendActionCommand(cfg.device_key, cfg.group_key, cfg.group_mask, target_ns,
                            /*requestAck=*/true, &result_size, results);

  std::printf("action_sync: scheduled AcquisitionStart for t=%luns (%.0fms from now)\n", target_ns,
              static_cast<double>(target_ns - now_ns) * 1e-6);
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
