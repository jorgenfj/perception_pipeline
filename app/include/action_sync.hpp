#pragma once

#include <Spinnaker.h>

#include <cstdint>

#include "app_config.hpp"
#include "spinnaker_source.hpp"

namespace perception {

// Validates a Scheduled Action Command actually landed where expected: the
// first observed timestamp against the schedule, and frame-to-frame spacing
// against 1/expected_hz. See spinnaker/README.md.
class ActionSyncChecker {
 public:
  // Called once per consumed frame; a no-op once enabled is false or the
  // verdict has already printed.
  void observe(uint64_t ts_ns);

  bool enabled = false;
  uint64_t target_ns = 0;
  double period_ns = 0.0;
  double tolerance_ms = 0.0;
  uint32_t check_frames = 0;
  // Sensor-specific GetTimeStamp() latch convention (e.g. end-of-exposure
  // instead of trigger instant), not part of the sync mechanism itself --
  // see action_sync.expected_start_offset_ms in app_config.hpp.
  double expected_start_offset_ms = 0.0;

 private:
  void print_verdict() const;

  uint32_t n = 0;
  uint64_t first_ts = 0;
  uint64_t prev_ts = 0;
  double spacing_min_ms = 1e30;
  double spacing_max_ms = -1e30;
  double spacing_sum_ms = 0.0;
  bool done = false;
};

// Schedules the AcquisitionStart Action Command that `cfg` describes (throws
// if the camera isn't PTP-Slave yet -- an unsynced camera acks NO_REF_TIME),
// prints the per-camera ack line, and arms `checker` to verify it landed.
// `checker` is a separate out-param rather than a return value: the caller
// already owns it and keeps observing frames against it for the rest of the
// run, long after this call returns.
void arm_action_sync(Spinnaker::SystemPtr& system, SpinnakerSource& source,
                      const ActionSyncConfig& cfg, ActionSyncChecker& checker);

}  // namespace perception
