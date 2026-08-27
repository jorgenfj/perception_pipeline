#include <Spinnaker.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "acquire_source.hpp"
#include "action_sync.hpp"
#include "frame_sink.hpp"  // host_now_ns()
#include "spinnaker_source.hpp"

namespace perception {
namespace {

// The camera-side setup a per-frame scheduled trigger needs, appended after the
// config's own features so it wins over whatever the yaml said. Order matters:
// keys and selector/source before TriggerMode On, and the frame-rate limiter
// off so the trigger paces acquisition rather than fighting it.
void append_frame_trigger_features(CameraConfig& camera, const ActionSyncConfig& sync) {
  auto set = [&](const char* node, std::string value) {
    camera.features.emplace_back(node, std::move(value));
  };
  set("ActionDeviceKey", std::to_string(sync.device_key));
  set("ActionGroupKey", std::to_string(sync.group_key));
  set("ActionGroupMask", std::to_string(sync.group_mask));
  set("AcquisitionFrameRateEnable", "false");
  set("TriggerSelector", "FrameStart");
  set("TriggerSource", "Action0");
  // No TriggerActivation: it selects an edge for Line triggers and is not
  // writable when the source is Action0 -- there is no edge on an action.
  set("TriggerMode", "On");
}

class SpinnakerAcquireSource final : public AcquireSource {
 public:
  explicit SpinnakerAcquireSource(const AppConfig& config)
      : system_(Spinnaker::System::GetInstance()), cameras_(system_->GetCameras()) {
    if (config.streams.empty()) throw std::runtime_error("no streams configured");

    std::vector<Spinnaker::CameraPtr> selected;
    for (const StreamConfig& stream : config.streams) {
      selected.push_back(SpinnakerSource::select(cameras_, stream.serial));
    }
    for (std::size_t a = 0; a < selected.size(); ++a) {
      for (std::size_t b = a + 1; b < selected.size(); ++b) {
        if (&*selected[a] != &*selected[b]) continue;
        throw std::runtime_error(
            "streams[" + std::to_string(a) + "] (" + config.streams[a].role + ") and streams[" +
            std::to_string(b) + "] (" + config.streams[b].role +
            ") both resolved to the same camera -- set both serials in `streams:`, or attach "
            "the second camera");
      }
    }

    for (std::size_t s = 0; s < config.streams.size(); ++s) {
      CameraConfig camera = config.camera;
      camera.serial = config.streams[s].serial;
      // Has to happen here, before the SpinnakerSource ctor applies the feature
      // list -- TriggerMode cannot be flipped once acquisition is running.
      if (config.action_sync.enabled && config.action_sync.per_frame) {
        append_frame_trigger_features(camera, config.action_sync);
      }
      streams_.push_back(Stream{config.streams[s].role, config.streams[s].serial,
                                std::make_unique<SpinnakerSource>(selected[s], camera)});
    }

    for (std::size_t s = 1; s < streams_.size(); ++s) {
      const CameraGeometry& a = streams_[0].source->geometry();
      const CameraGeometry& b = streams_[s].source->geometry();
      if (a.width != b.width || a.height != b.height || a.pixel_format != b.pixel_format) {
        throw std::runtime_error(
            "stream 0 (" + streams_[0].role + ") came up " + std::to_string(a.width) + "x" +
            std::to_string(a.height) + " " + a.pixel_format + " but stream " +
            std::to_string(s) + " (" + streams_[s].role + ") came up " +
            std::to_string(b.width) + "x" + std::to_string(b.height) + " " + b.pixel_format +
            " -- the geometry in camera.features applies to both, so this means one camera "
            "rejected part of it");
      }
    }
  }

  ~SpinnakerAcquireSource() override {
    stop_action_sync();
    streams_.clear();
    cameras_.Clear();
    system_->ReleaseInstance();
  }

  uint32_t stream_count() const override { return static_cast<uint32_t>(streams_.size()); }

  FrameSource& source(uint32_t stream) override { return *streams_.at(stream).source; }

  std::string describe(uint32_t stream) const override {
    const Stream& s = streams_.at(stream);
    const std::string what =
        s.serial.empty() ? std::string("camera (first found)") : "camera " + s.serial;
    return s.role + ": " + what;
  }

  void arm_action_sync(const ActionSyncConfig& config,
                       const std::vector<ActionSyncChecker*>& checkers) override {
    if (!config.enabled) return;

    if (config.per_frame) {
      start_frame_triggers(config, checkers);
      return;
    }

    std::vector<std::string> ptp;
    for (const Stream& s : streams_) ptp.push_back(s.source->ptp_status());

    perception::arm_action_sync(system_, ptp, config, *checkers.at(0));
    propagate_checkers(checkers);
  }

  void stop_action_sync() override {
    trigger_run_.store(false, std::memory_order_relaxed);
    if (trigger_thread_.joinable()) trigger_thread_.join();
  }

  std::string trigger_health_line() const override {
    if (!triggering_) return {};
    char line[160];
    std::snprintf(line, sizeof(line), "trig sent=%lu ok=%lu missing_ack=%lu not_ok=%lu",
                  static_cast<unsigned long>(trigger_sent_.load(std::memory_order_relaxed)),
                  static_cast<unsigned long>(trigger_ok_.load(std::memory_order_relaxed)),
                  static_cast<unsigned long>(trigger_missing_.load(std::memory_order_relaxed)),
                  static_cast<unsigned long>(trigger_not_ok_.load(std::memory_order_relaxed)));
    return line;
  }

 private:
  // Copies the arming result onto every other stream's checker and labels them.
  // One broadcast covers the rig, so the schedule is shared; only the label and
  // the frames each one sees differ.
  void propagate_checkers(const std::vector<ActionSyncChecker*>& checkers) {
    checkers.at(0)->label = streams_[0].role;
    for (std::size_t s = 1; s < checkers.size() && s < streams_.size(); ++s) {
      ActionSyncChecker& other = *checkers[s];
      other.enabled = checkers[0]->enabled;
      other.target_ns = checkers[0]->target_ns;
      other.period_ns = checkers[0]->period_ns;
      other.tolerance_ms = checkers[0]->tolerance_ms;
      other.check_frames = checkers[0]->check_frames;
      other.expected_start_offset_ms = checkers[0]->expected_start_offset_ms;
      other.label = streams_[s].role;
    }
  }

  // Polls until every camera reads PTP "Slave", or the timeout expires.
  // Applying the camera config resets the PTP state machine, so they are all in
  // "Listening" right now and need a few Announce intervals: sampling once and
  // giving up fails a rig that was about to work.
  bool wait_for_ptp_slave(std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::string last;
    bool announced = false;

    for (;;) {
      std::string joined;
      bool all_slave = true;
      for (const Stream& s : streams_) {
        const std::string status = s.source->ptp_status();
        if (status != "Slave") all_slave = false;
        if (!joined.empty()) joined += "/";
        joined += status.empty() ? "unsupported" : status;
      }
      if (all_slave) {
        if (announced) std::printf("ptp: locked\n");
        return true;
      }
      // Only on change: polling at 4 Hz would otherwise print forty identical
      // lines while nothing happens.
      if (joined != last) {
        std::printf("ptp: %s%s\n", joined.c_str(),
                    announced ? "" : " -- waiting for every camera to reach Slave");
        last = joined;
        announced = true;
      }
      if (std::chrono::steady_clock::now() >= deadline) return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
  }

  // Check the keys before sending, not after. A camera whose keys do not match
  // the broadcast does not report an error -- it just does not ack, and with
  // TriggerMode On it then waits forever for a trigger it will never accept.
  // The visible symptom is "no acks" plus delivered=0, which says nothing about
  // which of the three keys was wrong.
  void verify_action_keys(const ActionSyncConfig& config) {
    for (std::size_t id = 0; id < streams_.size(); ++id) {
      uint32_t device_key = 0, group_key = 0, group_mask = 0;
      const std::string who = "stream " + std::to_string(id) + " (" + streams_[id].role + ")";
      switch (streams_[id].source->action_keys(device_key, group_key, group_mask)) {
        case SpinnakerSource::ActionKeys::Absent:
          throw std::runtime_error(
              "action_sync: " + who +
              " exposes no ActionDeviceKey/ActionGroupKey/ActionGroupMask nodes -- this camera "
              "model does not support Action Commands, so per_frame triggering cannot work");
        case SpinnakerSource::ActionKeys::WriteOnly:
          // Nothing to compare against; the features were just applied from the
          // same config, so trust them rather than refusing to run.
          break;
        case SpinnakerSource::ActionKeys::Readable:
          if (device_key != config.device_key || group_key != config.group_key ||
              (group_mask & config.group_mask) == 0) {
            throw std::runtime_error(
                "action_sync: " + who + " has device_key=" + std::to_string(device_key) +
                " group_key=" + std::to_string(group_key) + " group_mask=" +
                std::to_string(group_mask) + " but the config broadcasts device_key=" +
                std::to_string(config.device_key) + " group_key=" +
                std::to_string(config.group_key) + " group_mask=" +
                std::to_string(config.group_mask) +
                " -- a command whose keys do not match is silently ignored by the camera");
          }
          break;
      }
    }
  }

  // One scheduled Action Command per frame: every exposure, not just the first,
  // is pinned to a PTP instant. See ActionSyncConfig::per_frame for why this is
  // not the default.
  void start_frame_triggers(const ActionSyncConfig& config,
                            const std::vector<ActionSyncChecker*>& checkers) {
    const double hz = config.trigger_hz > 0.0 ? config.trigger_hz : config.expected_hz;
    if (hz <= 0.0) {
      throw std::runtime_error(
          "action_sync: per_frame needs a positive trigger_hz (or expected_hz to fall back on)");
    }

    if (!wait_for_ptp_slave(std::chrono::milliseconds(config.ptp_wait_ms))) {
      // Not fatal the way the one-shot path is: the trigger loop keeps running
      // and starts working the moment PTP locks, so a slow grandmaster costs
      // the first few frames rather than the run.
      std::printf("ptp: still not locked after %ums -- triggers will NO_REF_TIME until it locks\n",
                  config.ptp_wait_ms);
    }
    verify_action_keys(config);

    const uint64_t period_ns = static_cast<uint64_t>(1e9 / hz);
    const uint64_t lead_ns = static_cast<uint64_t>(config.trigger_lead_ms * 1e6);
    const uint64_t first_target = host_now_ns() + lead_ns;
    const unsigned int ncam = static_cast<unsigned int>(streams_.size());

    std::printf("action_sync: per-frame scheduled triggers at %.2f Hz, %.0f ms lead "
                "(~%.0f commands in flight; OVERFLOW acks mean ActionQueueSize is smaller)\n",
                hz, config.trigger_lead_ms,
                period_ns ? static_cast<double>(lead_ns) / static_cast<double>(period_ns) : 0.0);

    // The checker validates spacing against the trigger rate, not
    // AcquisitionFrameRate -- that limiter is off in this mode.
    ActionSyncChecker& first = *checkers.at(0);
    first.enabled = true;
    first.target_ns = first_target;
    first.period_ns = static_cast<double>(period_ns);
    first.tolerance_ms = config.tolerance_ms;
    first.check_frames = config.check_frames;
    first.expected_start_offset_ms = config.expected_start_offset_ms;
    propagate_checkers(checkers);

    triggering_ = true;
    trigger_run_.store(true, std::memory_order_relaxed);
    trigger_thread_ = std::thread([this, config, period_ns, lead_ns, first_target, ncam] {
      uint64_t target = first_target;
      while (trigger_run_.load(std::memory_order_relaxed)) {
        // pResultSize is the EXPECTED device count, and the call blocks until
        // that many acks arrive or an internal timeout fires. It must be the
        // real camera count: passing an array capacity makes every call wait
        // out the timeout for devices that do not exist.
        Spinnaker::ActionCommandResult results[kMaxAckResults];
        unsigned int result_size = ncam;
        system_->SendActionCommand(config.device_key, config.group_key, config.group_mask, target,
                                   /*requestAck=*/true, &result_size, results);
        trigger_sent_.fetch_add(1, std::memory_order_relaxed);

        unsigned int ok = 0;
        for (unsigned int i = 0; i < result_size; ++i) {
          if (results[i].Status == Spinnaker::SPINNAKER_ACTION_COMMAND_STATUS_OK) ++ok;
        }
        if (result_size < ncam) trigger_missing_.fetch_add(1, std::memory_order_relaxed);
        if (ok >= ncam) {
          trigger_ok_.fetch_add(1, std::memory_order_relaxed);
        } else {
          trigger_not_ok_.fetch_add(1, std::memory_order_relaxed);
        }

        target += period_ns;
        const int64_t wake = static_cast<int64_t>(target) - static_cast<int64_t>(lead_ns);
        const int64_t delta = wake - static_cast<int64_t>(host_now_ns());
        if (delta > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(delta));
      }
    });
  }

  // Ack array capacity. Two streams is the configured maximum, but a stray
  // camera on the same keys can ack too, and the SDK writes as many as arrive.
  static constexpr unsigned int kMaxAckResults = 8;

  struct Stream {
    std::string role;
    std::string serial;
    std::unique_ptr<SpinnakerSource> source;
  };

  Spinnaker::SystemPtr system_;
  Spinnaker::CameraList cameras_;
  std::vector<Stream> streams_;

  // Per-frame triggering. `triggering_` is set once on the arming thread before
  // the loop starts and read afterwards, so it needs no atomicity; the counters
  // are written by the trigger thread and read by the reporter.
  bool triggering_ = false;
  std::thread trigger_thread_;
  std::atomic<bool> trigger_run_{false};
  std::atomic<uint64_t> trigger_sent_{0};
  std::atomic<uint64_t> trigger_ok_{0};
  std::atomic<uint64_t> trigger_missing_{0};
  std::atomic<uint64_t> trigger_not_ok_{0};
};

}  // namespace

std::unique_ptr<AcquireSource> make_acquire_source(const AppConfig& config) {
  return std::make_unique<SpinnakerAcquireSource>(config);
}

const char* acquire_source_kind() { return "spinnaker"; }

}  // namespace perception
