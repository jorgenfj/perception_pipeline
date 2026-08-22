#include "action_sync_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace perception {

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
      "\naction_sync%s%s: %s -- start offset %.3f ms, expected %.3f ms, deviation %.3f ms "
      "(tolerance %.2f ms); frame spacing deviation from 1/expected_hz min/mean/max "
      "%.3f/%.3f/%.3f ms over %u frames\n",
      label.empty() ? "" : " ", label.c_str(), (start_ok && spacing_ok) ? "PASS" : "FAIL",
      start_offset_ms, expected_start_offset_ms, start_deviation_ms, tolerance_ms, spacing_min_ms,
      spacing_mean_ms, spacing_max_ms, n);
}

}  // namespace perception
