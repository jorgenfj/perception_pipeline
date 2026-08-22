#pragma once

#include <cstdint>
#include <string>

namespace perception {

// Checks where a Scheduled Action Command's frames actually landed: the first
// observed timestamp against the schedule, and frame-to-frame spacing against
// 1/expected_hz. See spinnaker/README.md.
//
// Deliberately free of <Spinnaker.h>, so a build with no SDK can still hold one
// -- sending the command needs the vendor library, judging the result does not.
//
class ActionSyncChecker {
 public:
  // Called once per delivered frame; a no-op once enabled is false or the
  // verdict has already printed.
  void observe(uint64_t ts_ns);

  bool enabled = false;
  uint64_t target_ns = 0;
  double period_ns = 0.0;
  double tolerance_ms = 0.0;
  uint32_t check_frames = 0;
  // Sensor-specific GetTimeStamp() latch convention (e.g. end-of-exposure
  // instead of trigger instant), not part of the sync mechanism itself --
  // see ActionSyncConfig::expected_start_offset_ms in camera_config.hpp.
  double expected_start_offset_ms = 0.0;

  // Labels the verdict line, so a stereo rig's two checkers are tellable apart.
  std::string label;

  // First camera timestamp seen. The stereo caller compares the two cameras'
  // values against each other: their difference is how far apart the two eyes
  // *claim* to have started, measured from the single instant both were given.
  bool have_first() const { return n > 0; }
  uint64_t first_timestamp_ns() const { return first_ts; }
  bool finished() const { return done; }

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

}  // namespace perception
