#pragma once

#include <cstdint>
#include <string>

#include "app_config.hpp"
#include "device_ring_buffer.hpp"
#include "latency_probe.hpp"
#include "ring_lease.hpp"
#include "spinnaker_source.hpp"
#include "upload_stage.hpp"

namespace perception {

struct LatencyWindow {
  double min_ms = 1e30;
  double max_ms = -1e30;
  double sum_ms = 0.0;
  uint64_t count = 0;

  void add(double ms);
  double mean_ms() const { return count ? sum_ms / static_cast<double>(count) : 0.0; }
  void reset() { *this = LatencyWindow{}; }
};

// Ongoing (not one-shot) frame-spacing check against action_sync.expected_hz
// -- runs for the whole session, reported every 60 frames alongside the rest
// of the health line, so a rate that drifts or degrades after
// ActionSyncChecker's own one-shot verdict still shows up.
struct SpacingWindow {
  uint64_t prev_ts = 0;
  bool have_prev = false;
  double min_ms = 1e30;
  double max_ms = -1e30;
  double sum_ms = 0.0;
  uint64_t count = 0;

  void add(uint64_t ts_ns, double period_ns);
  double mean_ms() const { return count ? sum_ms / static_cast<double>(count) : 0.0; }
  // Actual measured rate implied by the mean spacing this window, not just
  // the deviation from expected -- lets a report line stand on its own.
  double actual_hz(double period_ns) const;
  // have_prev/prev_ts survive a reset deliberately: spacing across the window
  // boundary is still a real gap between two real frames and should still be
  // measured, not dropped on the floor.
  void reset();
};

// Owns the running-report bookkeeping (latency/spacing accumulation between
// lines) and prints the periodic health line plus the final summary. Keeps
// the main loop's body to "lease a frame, hand it to the reporter" rather
// than a page of printf plumbing.
class Reporter {
 public:
  // `ptp_status_at_start` is the one-shot Init()-time read the caller already
  // did to print the startup line; kept here only to gate whether the ptp
  // segment ever prints -- a camera without PTP support stays empty forever.
  Reporter(SpinnakerSource& source, UploadStage& upload, DeviceRingBuffer& device_ring,
           LatencyProbe& probe, const AppConfig& config, std::string ptp_status_at_start);

  // Call once per consumed frame; prints a line every 60th call.
  void observe(uint64_t consumed, const ReadLease& lease);

  // Final counters at shutdown, in the same shape as the periodic line.
  void print_summary(uint64_t consumed) const;

 private:
  static constexpr uint64_t kReportEvery = 60;

  void print(uint64_t consumed, const ReadLease& lease);

  SpinnakerSource* source_;
  UploadStage* upload_;
  DeviceRingBuffer* device_ring_;
  LatencyProbe* probe_;
  const AppConfig* config_;
  std::string ptp_status_at_start_;

  LatencyWindow latency_;
  SpacingWindow fps_window_;
  double action_sync_period_ns_;
};

}  // namespace perception
