#include "report.hpp"

#include <algorithm>
#include <cstdio>
#include <string>

namespace perception {

void LatencyWindow::add(double ms) {
  min_ms = std::min(min_ms, ms);
  max_ms = std::max(max_ms, ms);
  sum_ms += ms;
  ++count;
}

void SpacingWindow::add(uint64_t ts_ns, double period_ns) {
  if (have_prev) {
    const double gap_ms = static_cast<double>(ts_ns - prev_ts) * 1e-6;
    const double dev_ms = gap_ms - period_ns * 1e-6;
    min_ms = std::min(min_ms, dev_ms);
    max_ms = std::max(max_ms, dev_ms);
    sum_ms += dev_ms;
    ++count;
  }
  prev_ts = ts_ns;
  have_prev = true;
}

double SpacingWindow::actual_hz(double period_ns) const {
  const double mean_period_ms = period_ns * 1e-6 + mean_ms();
  return mean_period_ms > 0.0 ? 1000.0 / mean_period_ms : 0.0;
}

void SpacingWindow::reset() {
  min_ms = 1e30;
  max_ms = -1e30;
  sum_ms = 0.0;
  count = 0;
}

Reporter::Reporter(FrameSource& source, UploadStage& upload, DeviceRingBuffer& device_ring,
                   LatencyProbe& probe, const AppConfig& config, std::string ptp_status_at_start)
    : source_(&source),
      upload_(&upload),
      device_ring_(&device_ring),
      probe_(&probe),
      config_(&config),
      ptp_status_at_start_(std::move(ptp_status_at_start)),
      action_sync_period_ns_(1e9 / config.action_sync.expected_hz) {}

void Reporter::observe(uint64_t consumed, const ReadLease& lease) {
  double latency_ms = -1.0;
  int64_t processing_ns = 0;
  if (probe_->processing_ns(lease.timestamp_ns(), processing_ns)) {
    latency_ms = static_cast<double>(processing_ns) * 1e-6;
    latency_.add(latency_ms);
  }

  if (config_->action_sync.enabled) fps_window_.add(lease.timestamp_ns(), action_sync_period_ns_);

  if (consumed % kReportEvery == 0) print(consumed, lease);
}

void Reporter::print(uint64_t consumed, const ReadLease& lease) {
  std::printf("t=%luns slot=%u delivered=%lu uploaded=%lu consumed=%lu ", lease.timestamp_ns(),
              lease.slot(), source_->delivered(), upload_->uploaded(), consumed);
  const std::string counters = source_->counters();
  if (!counters.empty()) std::printf("%s ", counters.c_str());
  std::printf("stalls=%lu dropped=%lu failed=%lu", device_ring_->write_stalls(),
              upload_->dropped(), upload_->failed());
  const std::string notes = source_->notes();
  if (!notes.empty()) std::printf(" %s", notes.c_str());

  double raw_min_ms = 0.0, raw_mean_ms = 0.0, raw_max_ms = 0.0;
  if (probe_->raw_snapshot_reset(raw_min_ms, raw_mean_ms, raw_max_ms)) {
    std::printf(" | cam->host min/mean/max %.2f/%.2f/%.2f ms", raw_min_ms, raw_mean_ms, raw_max_ms);
  }
  if (latency_.count) {
    std::printf(" | host->device min/mean/max %.2f/%.2f/%.2f ms", latency_.min_ms,
                latency_.mean_ms(), latency_.max_ms);
  }
  if (config_->action_sync.enabled && fps_window_.count > 0) {
    std::printf(" | fps actual=%.4f expected=%.4f spacing dev min/mean/max %.3f/%.3f/%.3f ms",
                fps_window_.actual_hz(action_sync_period_ns_), config_->action_sync.expected_hz,
                fps_window_.min_ms, fps_window_.mean_ms(), fps_window_.max_ms);
    fps_window_.reset();
  }
  if (!ptp_status_at_start_.empty()) {
    int64_t offset_ns = 0;
    const std::string ptp_now = source_->ptp_status();
    std::printf(" | ptp %s", ptp_now.c_str());
    if (ptp_now == "Slave" && source_->ptp_offset_ns(offset_ns)) {
      std::printf(" offset=%ldns", offset_ns);
    }
  }
#ifdef PERCEPTION_TRACE_POOL
  const std::string pool = source_->pool_line(config_->pipeline.ingress_depth);
  if (!pool.empty()) std::printf(" | %s", pool.c_str());
#endif
  std::printf("\n");
  latency_.reset();
}

void Reporter::print_summary(uint64_t consumed) const {
  std::printf("\ndelivered=%lu uploaded=%lu consumed=%lu ", source_->delivered(),
              upload_->uploaded(), consumed);
  const std::string counters = source_->counters();
  if (!counters.empty()) std::printf("%s ", counters.c_str());
  std::printf("stalls=%lu dropped=%lu failed=%lu\n", device_ring_->write_stalls(),
              upload_->dropped(), upload_->failed());
  const std::string notes = source_->notes();
  if (!notes.empty()) std::printf("%s\n", notes.c_str());
#ifdef PERCEPTION_TRACE_POOL
  const std::string pool = source_->pool_line(config_->pipeline.ingress_depth);
  if (!pool.empty()) std::printf("%s\n", pool.c_str());
#endif
}

}  // namespace perception
