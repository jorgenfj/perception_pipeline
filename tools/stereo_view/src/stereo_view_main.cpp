//
//   stereo_view [options] [config.yaml]
//
// Two cameras -> heap buffers -> host pairing by camera timestamp -> a window
// with both halves side by side and the pair's skew under them.
//
// The point of it being CUDA-free is that the questions this answers -- are the
// cameras synced, are they pointed at the same thing, is the exposure right, is
// PTP actually locked -- are the questions you have before the GPU pipeline is
// worth starting, and often on a machine that has no GPU in it at all. The
// Spinnaker SDK it does need: this tool opens cameras.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "camera_config.hpp"
#include "config_path.hpp"
#include "cpu_debayer.hpp"
#include "stereo_config.hpp"
#include "stereo_live.hpp"
#include "stereo_pairer.hpp"

#ifdef OPENGL_DISPLAY
#include "stereo_view.hpp"
#endif

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

// This tool's own config directory, the fallback when there is no config/ next
// to the binary. See capture/include/config_path.hpp.
#ifndef PERCEPTION_CONFIG_DIR
#define PERCEPTION_CONFIG_DIR ""
#endif

std::string config_file(const char* name) {
  return perception::resolve_config_path(name, PERCEPTION_CONFIG_DIR);
}

// The two fields LiveStereo needs out of the config this tool reads. See the
// note on StereoCameras: the SDK wrapper does not take a viewer's config.
perception::StereoCameras stereo_cameras(const perception::StereoConfig& config) {
  perception::StereoCameras cameras;
  for (uint32_t id = 0; id < 2; ++id) cameras.serials[id] = config.streams[id].serial;
  cameras.buffer_count = config.buffer_count;
  return cameras;
}

// Put the cameras in per-frame FrameStart / Action0 trigger mode so each
// scheduled Action Command fires one synchronized capture. Appended after the
// config's own features so it wins over the yaml's free-run TriggerMode. Order:
// keys and selector/source before TriggerMode On, frame-rate limiter off so the
// trigger alone paces acquisition.
void append_per_frame_trigger(perception::CameraConfig& camera,
                              const perception::ActionSyncConfig& sync) {
  auto set = [&](const char* node, std::string v) { camera.features.emplace_back(node, v); };
  set("ActionDeviceKey", std::to_string(sync.device_key));
  set("ActionGroupKey", std::to_string(sync.group_key));
  set("ActionGroupMask", std::to_string(sync.group_mask));
  set("AcquisitionFrameRateEnable", "false");
  set("TriggerSelector", "FrameStart");
  set("TriggerSource", "Action0");
  set("TriggerMode", "On");
}

struct Options {
  std::string config_path;
  bool no_display = false;
  uint32_t decimate = 0;   // 0 means "whatever the config says"
  uint64_t tolerance_us = 0;
  uint64_t max_pairs = 0;
  bool sync_start = false;
  uint32_t ptp_watch_s = 0;
  double capture_hz = 0.0;  // >0: drive per-frame scheduled triggers at this rate
};

void print_usage() {
  std::printf(
      "stereo_view [options] [config.yaml]\n"
      "\n"
      "  --no-display      no window -- report only\n"
      "  --decimate N      Bayer block per displayed pixel: 2, 4 or 8\n"
      "  --tolerance-us N  pairing tolerance, overriding the config\n"
      "  --pairs N         stop after N pairs\n"
      "  --sync-start      schedule a PTP AcquisitionStart on both cameras and check\n"
      "                    where it landed (needs both cameras PTP Slave)\n"
      "  --capture-hz N    drive per-frame PTP-scheduled triggers at N fps so EVERY\n"
      "                    exposure is aligned (not just the start). Watch max_skew --\n"
      "                    this is what stops the two shutters drifting apart\n"
      "  --ptp-watch N     open both cameras WITHOUT streaming and watch their PTP\n"
      "                    lock for N seconds, then exit. Run it idle, then again\n"
      "                    while streaming, to see if the traffic is what breaks it\n"
      "\n"
      "  In the window: space pauses, q or esc quits.\n");
}

// Argument parsing that reports what it did not understand rather than ignoring
// it -- a silently dropped --capture-hz is a run you have to do twice.
Options parse_args(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string(name) + " needs a value");
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      print_usage();
      std::exit(0);
    } else if (arg == "--no-display") {
      options.no_display = true;
    } else if (arg == "--decimate") {
      options.decimate = static_cast<uint32_t>(std::stoul(value("--decimate")));
    } else if (arg == "--tolerance-us") {
      options.tolerance_us = std::stoull(value("--tolerance-us"));
    } else if (arg == "--ptp-watch") {
      options.ptp_watch_s = static_cast<uint32_t>(std::stoul(value("--ptp-watch")));
    } else if (arg == "--sync-start") {
      options.sync_start = true;
    } else if (arg == "--capture-hz") {
      options.capture_hz = std::stod(value("--capture-hz"));
    } else if (arg == "--pairs") {
      options.max_pairs = std::stoull(value("--pairs"));
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option " + arg);
    } else {
      options.config_path = arg;
    }
  }
  return options;
}

// What the two cameras' GevIEEE1588Status means, said out loud.
//
// This is the first thing that goes wrong on a rig and the last thing anyone
// checks, so it gets a paragraph rather than a word in a status line. Empty
// when both cameras are Slave, i.e. when there is nothing to warn about.
std::string ptp_verdict(const std::vector<std::string>& statuses) {
  if (statuses.size() != 2) return "";
  if (statuses[0] == "Slave" && statuses[1] == "Slave") return "";

  if (statuses[0].empty() || statuses[1].empty()) {
    return "ptp: a camera does not expose GevIEEE1588 at all. With no shared clock the two\n"
           "     timestamp streams are unrelated counters and NOTHING WILL PAIR.\n";
  }

  if (statuses[0] == "Master" && statuses[1] == "Master") {
    return "ptp: BOTH CAMERAS ARE MASTER -- neither heard a grandmaster, so each elected\n"
           "     itself. There is no common epoch, their timestamps are unrelated counters,\n"
           "     and NOTHING WILL PAIR. Start ptp4l on this host; with no PHC on the NIC\n"
           "     that is the software-master form (see spinnaker/README.md):\n"
           "       sudo ptp4l -i <iface> -S -f /etc/ptp4l-master.conf -m\n";
  }

  return "ptp: not locked yet. BMC needs a few Announce intervals to settle, so this is\n"
         "     normal for the first few seconds. If it stays here, nothing will pair.\n";
}

// Watches both cameras' PTP state without streaming a single frame.
//
// The question this answers is the one a status line cannot: does the lock hold
// when the rig is idle? Acquisition is never started, so there is no image
// traffic on the link and no second thread on the node map. Run it idle, then
// run it again while stereo_view streams in another terminal -- if the lock is
// steady idle and flaps under load, the cause is the load (a saturated uplink
// starving Announce and Delay_Req, or the host too busy to send Announce on
// time), not the PTP configuration.
class PtpWatch {
 public:
  void observe(const std::vector<perception::LiveStereo::PtpSample>& samples) {
    ++samples_;
    const bool locked = samples.size() == 2 && samples[0].status == "Slave" &&
                        samples[1].status == "Slave";
    // Run lengths on both sides of the transition, because how the lock fails
    // matters more than how often. Random loss gives ragged runs; a regular
    // cycle means something is timing out on a schedule.
    if (samples_ > 1 && locked != was_locked_) {
      (was_locked_ ? locked_runs_ : unlocked_runs_).push_back(run_);
      run_ = 0;
    }
    ++run_;
    was_locked_ = locked;

    if (locked) {
      ++locked_samples_;
      longest_run_ = std::max(longest_run_, run_);
      for (const auto& sample : samples) {
        if (!sample.has_offset) continue;
        const int64_t magnitude = sample.offset_ns < 0 ? -sample.offset_ns : sample.offset_ns;
        worst_offset_ns_ = std::max(worst_offset_ns_, magnitude);
      }
    }

    // Only transitions, not every sample: at 5 Hz the per-sample form is 150
    // lines of noise for a 30s run, and the edges are the whole signal anyway.
    if (samples_ > 1 && locked == printed_locked_) return;
    printed_locked_ = locked;

    char line[224];
    std::snprintf(line, sizeof(line), "  +%6.2fs  ",
                  static_cast<double>(samples_ - 1) * interval_s_);
    std::string out = line;
    for (std::size_t i = 0; i < samples.size(); ++i) {
      char field[96];
      if (samples[i].has_offset && samples[i].status == "Slave") {
        std::snprintf(field, sizeof(field), "%scam%zu %-9s offset=%+ldns", i ? "  " : "", i,
                      samples[i].status.c_str(), static_cast<long>(samples[i].offset_ns));
      } else {
        std::snprintf(field, sizeof(field), "%scam%zu %-9s", i ? "  " : "", i,
                      samples[i].status.empty() ? "unsupported" : samples[i].status.c_str());
      }
      out += field;
    }
    std::printf("%s\n", out.c_str());
  }

  void report(double seconds) const {
    if (samples_ == 0) return;
    const double locked_fraction =
        static_cast<double>(locked_samples_) / static_cast<double>(samples_);

    std::printf("\nptp over %.0fs: both Slave for %.0f%% of samples, longest unbroken lock %.2fs",
                seconds, locked_fraction * 100.0,
                static_cast<double>(longest_run_) * interval_s_);
    if (worst_offset_ns_ > 0) {
      std::printf(", worst reported offset %ldns", static_cast<long>(worst_offset_ns_));
    }
    std::printf("\n");

    if (locked_fraction >= 0.99) {
      std::printf(
          "The lock is steady with the rig idle. Run this again while stereo_view is\n"
          "streaming: if it starts flapping then, the traffic is what breaks PTP, and the\n"
          "fix is headroom -- lower the frame rate, or set DeviceLinkThroughputLimit to\n"
          "around 60%% of what the link can carry, so Announce and Delay_Req are not\n"
          "queued behind a frame.\n");
      return;
    }

    if (!locked_runs_.empty() && !unlocked_runs_.empty()) {
      std::printf("     cycle: %.2fs locked / %.2fs unlocked%s\n",
                  mean(locked_runs_) * interval_s_, mean(unlocked_runs_) * interval_s_,
                  is_regular() ? ", repeating" : ", irregular");
    }

    std::printf(
        "\nThe lock is NOT holding with the rig idle -- no image traffic, no acquisition\n"
        "thread -- so this is not about link saturation or host load.\n");

    // Ordered by how often each one turns out to be the answer, with the one
    // this build can actually detect first.
    //
    // Deliberately NOT led by announceReceiptTimeout. That was the first guess
    // here and it was wrong: raising the Announce rate did not stabilise the
    // lock, the flap period just tracked it 1:1. A timeout expiring would have
    // become *more* stable with more Announces, so a cycle that follows the
    // Announce interval is evidence against that explanation, not for it.
    if (ntp_daemon_running()) {
      std::printf(
          "\nsystemd-timesyncd IS RUNNING on this host, and that is almost certainly it.\n"
          "In software master mode (ptp4l -S, no PHC) ptp4l serves CLOCK_REALTIME -- it\n"
          "announces the host's own system clock as the PTP timebase. timesyncd is\n"
          "disciplining that same clock from an NTP server at the same time, so the\n"
          "cameras are handed a timebase that something else keeps stepping underneath\n"
          "them. They lock, see the discontinuity, and drop.\n"
          "  sudo timedatectl set-ntp false\n"
          "  sudo systemctl restart ptp4l          # or restart it by hand\n"
          "Then re-run this. CLOCK_REALTIME free-runs afterwards, which is what\n"
          "gm.ClockClass 248 already says and is fine for stereo pairing: both cameras\n"
          "share the host's epoch whatever it is. If absolute UTC matters across\n"
          "sessions, that is the case for a GPS-disciplined grandmaster instead.\n");
    }

    std::printf(
        "\nIf that is not it, in order:\n"
        "  1. Multicast delivery. PTP is multicast (224.0.1.129); a switch doing IGMP\n"
        "     snooping with no querier stops forwarding it once membership ages out.\n"
        "  2. Watch the wire and time the gaps directly:\n"
        "       sudo tcpdump -i <iface> -n 'udp port 319 or udp port 320'\n"
        "     Announce from the host should be regular. A camera sending its own\n"
        "     Announce has given up hearing one, and the timestamp of that packet is\n"
        "     when it gave up.\n");
  }

  // Never locked at all is a different problem from a lock that comes and goes.
  bool ever_locked() const { return locked_samples_ > 0; }

  void set_interval(double seconds) { interval_s_ = seconds; }

 private:
  // systemd-timesyncd drops this file once it has synchronised, so its presence
  // means something other than ptp4l is steering CLOCK_REALTIME -- the clock
  // ptp4l is busy announcing to the cameras in software master mode.
  static bool ntp_daemon_running() {
    std::error_code ec;
    return std::filesystem::exists("/run/systemd/timesync/synchronized", ec);
  }

  static double mean(const std::vector<uint32_t>& runs) {
    if (runs.empty()) return 0.0;
    double total = 0.0;
    for (uint32_t run : runs) total += run;
    return total / static_cast<double>(runs.size());
  }

  // "Regular" at 1 Hz sampling means every completed run is within one sample
  // of its mean -- enough to tell a timeout firing on a schedule from packets
  // going missing, which is the whole distinction that picks the fix.
  bool is_regular() const {
    if (locked_runs_.size() < 2 || unlocked_runs_.size() < 2) return false;
    for (const std::vector<uint32_t>* runs : {&locked_runs_, &unlocked_runs_}) {
      const double average = mean(*runs);
      for (uint32_t run : *runs) {
        if (std::fabs(static_cast<double>(run) - average) > 1.0) return false;
      }
    }
    return true;
  }

  uint32_t samples_ = 0;
  uint32_t locked_samples_ = 0;
  uint32_t run_ = 0;
  uint32_t longest_run_ = 0;
  bool was_locked_ = false;
  bool printed_locked_ = false;
  int64_t worst_offset_ns_ = 0;
  double interval_s_ = 1.0;
  std::vector<uint32_t> locked_runs_;
  std::vector<uint32_t> unlocked_runs_;
};

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);

  // Line-buffered even when stdout is a pipe or a log file. The default there
  // is a 4 KB block buffer, which is enough to swallow every startup
  // diagnostic this tool prints -- the geometry, the PTP verdict, the
  // action_sync acks -- and then lose all of it when the run is ended with
  // ctrl-C. Those lines are the entire point of running it.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  try {
    Options options = parse_args(argc, argv);

    const std::string config_path =
        options.config_path.empty() ? config_file("stereo.yaml") : options.config_path;
    perception::StereoConfig config = perception::load_stereo_config(config_path);
    perception::CameraConfig camera_config = perception::load_camera_config(config_path);
    perception::ActionSyncConfig action_sync_config =
        perception::load_action_sync_config(config_path);
    if (options.sync_start) action_sync_config.enabled = true;
    // Per-frame triggering owns acquisition timing, so it is incompatible with
    // the one-shot AcquisitionStart of --sync-start: the cameras go into
    // FrameStart trigger mode and only this loop makes them expose.
    if (options.capture_hz > 0.0) {
      append_per_frame_trigger(camera_config, action_sync_config);
      action_sync_config.enabled = false;  // not the one-shot path
    }
    std::printf("config: %s\n", config_path.c_str());

    if (options.decimate != 0) config.decimate = options.decimate;
    if (options.tolerance_us != 0) config.tolerance_ns = options.tolerance_us * 1000;
    if (options.no_display) config.display = false;

    // --ptp-watch opens the cameras and stops there: no ring, no pairer, no
    // window. Anything else on this path would put traffic on the link, which
    // is the one thing this measurement must not do.
    if (options.ptp_watch_s != 0) {
      perception::LiveStereo watch(stereo_cameras(config), camera_config,
                                   [](uint32_t, const perception::FrameMeta&, const void*) {});

      // 5 Hz, not 1 Hz. The effect being measured turned out to have a period
      // of about a second, and sampling that at 1 Hz is at or under Nyquist --
      // the first version of this reported a clean "1.0s locked / 1.0s
      // unlocked" that was partly its own aliasing. Sampling well above the
      // phenomenon is the difference between measuring it and beating with it.
      constexpr double kIntervalS = 0.2;
      const auto interval = std::chrono::milliseconds(200);
      const uint32_t ticks =
          static_cast<uint32_t>(static_cast<double>(options.ptp_watch_s) / kIntervalS);

      std::printf("ptp watch: cameras open, acquisition NOT started, %us at %.0f Hz\n",
                  options.ptp_watch_s, 1.0 / kIntervalS);

      PtpWatch stats;
      stats.set_interval(kIntervalS);
      const auto started = std::chrono::steady_clock::now();
      for (uint32_t i = 0; i < ticks && !g_stop; ++i) {
        stats.observe(watch.ptp_sample());
        std::this_thread::sleep_for(interval);
      }
      const double elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
      stats.report(elapsed_s);
      return stats.ever_locked() ? 0 : 1;
    }

    perception::StereoPairer::Config pairer_config;
    pairer_config.tolerance_ns = config.tolerance_ns;
    pairer_config.queue_frames = config.queue_frames;
    pairer_config.hold = std::chrono::milliseconds(config.hold_ms);
    perception::StereoPairer pairer(pairer_config);

    // --- source ---------------------------------------------------------------
    std::unique_ptr<perception::LiveStereo> live(new perception::LiveStereo(
        stereo_cameras(config), camera_config,
        [&pairer](uint32_t stream, const perception::FrameMeta& meta, const void* data) {
          pairer.push(stream, meta.timestamp_ns, meta.host_recv_ns, meta.frame_id, data,
                      meta.bytes);
        }));

    const perception::CameraGeometry& geometry = live->geometry();
    const uint32_t width = geometry.width;
    const uint32_t height = geometry.height;
    const uint32_t stride = geometry.stride_bytes;
    const std::string pixel_format = geometry.pixel_format;

    const std::vector<std::string> ptp_statuses = live->ptp_status();
    std::string ptp;
    for (const std::string& status : ptp_statuses) {
      if (!ptp.empty()) ptp += "/";
      ptp += status.empty() ? "unsupported" : status;
    }
    std::printf("cameras: %ux%u stride=%u %s, %zu bytes/frame, ptp=%s\n", geometry.width,
                geometry.height, geometry.stride_bytes, geometry.pixel_format.c_str(),
                geometry.frame_bytes, ptp.c_str());
    std::printf("%s", ptp_verdict(ptp_statuses).c_str());

    perception::HostPixelFormat format{};
    if (!perception::host_pixel_format_from_genicam(pixel_format, format)) {
      throw std::runtime_error("stereo: cannot display pixel format '" + pixel_format +
                               "' -- this viewer draws Mono8 and the four Bayer8 orders");
    }

    // --- window ---------------------------------------------------------------
#ifdef OPENGL_DISPLAY
    std::unique_ptr<perception::StereoView> view;
    if (config.display) {
      perception::StereoView::Config view_config;
      view_config.window_width = config.window_width;
      view_config.window_height = config.window_height;
      view_config.vsync = config.vsync;
      view_config.skew_scale_ns = config.tolerance_ns;
      try {
        view.reset(new perception::StereoView("stereo :: live", view_config));
        std::printf("display: %ux%u decimate=%u (space pauses, q quits)\n",
                    config.window_width, config.window_height, config.decimate);
      } catch (const std::exception& e) {
        std::printf("display: disabled (%s)\n", e.what());
      }
    }
#else
    if (config.display) {
      std::printf("display: compiled out (OPENGL_DISPLAY=OFF)\n");
    }
#endif

    // --- run ------------------------------------------------------------------
    live->start();

    // A scheduled AcquisitionStart, if the config asked for one. Armed after
    // start(): with TriggerMode=On/AcquisitionStart the cameras deliver nothing
    // until it fires, so BeginAcquisition has to have run first.
    //
    // Note what this does and does not buy. It aligns the *start* of both
    // cameras to one PTP instant; from there each free-runs on its own crystal
    // and it is ongoing PTP discipline, not this command, that keeps them
    // together. And it is checked against the cameras' own timestamps, so it
    // confirms the trigger landed -- not that the clocks telling you so are
    // right. See the note printed at the end of the run.
    perception::ActionSyncChecker sync_check[2];
    if (live && action_sync_config.enabled) {
      live->arm_scheduled_start(action_sync_config, sync_check[0], sync_check[1]);
    }

    // Per-frame synchronized capture: one PTP-scheduled trigger per frame, so
    // EVERY exposure fires on the shared clock, not just the start. This is what
    // stops the two shutters drifting apart -- watch max_skew stay small.
    std::atomic<bool> trig_run{true};
    std::atomic<uint64_t> trig_sent{0};
    std::atomic<uint64_t> trig_failed{0};
    std::thread trigger_thread;
    if (live && options.capture_hz > 0.0) {
      std::printf("capture: per-frame scheduled triggers at %.1f Hz (TAI-UTC=%lds)\n",
                  options.capture_hz, static_cast<long>(perception::tai_offset_s()));
      if (!live->wait_for_ptp_slave(std::chrono::milliseconds(action_sync_config.ptp_wait_ms))) {
        std::printf("capture: PTP not locked -- triggers will NO_REF_TIME until it locks\n");
      }
      if (!perception::ptp_timebase_ready()) {
        std::printf("capture: warning -- the kernel holds no TAI offset, so triggers are being "
                    "scheduled in UTC and every one will come back ACTION_LATE. Is phc2sys "
                    "running?\n");
      }
      trigger_thread = std::thread([&] {
        const uint64_t period_ns = static_cast<uint64_t>(1e9 / options.capture_hz);
        const uint64_t lead_ns = 100'000'000ull;  // schedule each trigger 100 ms ahead
        // PTP (TAI) throughout -- target and sleep alike. The camera judges the
        // scheduled instant against its own PTP clock, and host_now_ns() is UTC,
        // so building the target from it puts every command ~37 s in the past.
        uint64_t target = perception::ptp_now_ns() + lead_ns;
        while (trig_run.load(std::memory_order_relaxed) && !g_stop) {
          if (!live->send_trigger(action_sync_config, target))
            trig_failed.fetch_add(1, std::memory_order_relaxed);
          trig_sent.fetch_add(1, std::memory_order_relaxed);
          target += period_ns;
          const int64_t wake = static_cast<int64_t>(target) - static_cast<int64_t>(lead_ns);
          const int64_t d = wake - static_cast<int64_t>(perception::ptp_now_ns());
          if (d > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(d));
        }
      });
    }

    perception::HostImage image[2];
    bool paused = false;
    uint64_t emitted = 0;
    uint64_t last_report_emitted = 0;
    uint64_t pairs = 0;
    auto last_report = std::chrono::steady_clock::now();

    while (!g_stop) {
      if (live && live->failed()) {
        std::printf("stereo: a camera gave up, stopping\n");
        break;
      }
      if (options.max_pairs != 0 && pairs >= options.max_pairs) break;

#ifdef OPENGL_DISPLAY
      if (view) {
        // Blocks on the display connection, so an idle window costs nothing.
        // A new pair does not wake it; the ring is checked on the same tick,
        // which is what the short timeout is for.
        view->poll_wait(0.004);
        if (view->should_close()) break;
        if (view->take_pause_pressed()) {
          paused = !paused;
          std::printf("%s\n", paused ? "paused" : "running");
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
#else
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
#endif

      perception::StereoPairer::Pair pair;
      const bool have_pair = !paused && pairer.try_pop(pair);
      if (have_pair) {
        ++emitted;
        if (pair.complete()) ++pairs;
        for (uint32_t s = 0; s < 2; ++s) {
          if (pair.have[s]) sync_check[s].observe(pair.timestamp_ns[s]);
        }
#ifdef OPENGL_DISPLAY
        if (view) {
          for (uint32_t s = 0; s < 2; ++s) {
            if (!pair.have[s]) continue;
            perception::debayer_to_rgb(pair.data[s], width, height, stride, format,
                                       config.decimate, image[s]);
          }
          perception::StereoView::Status status;
          status.have[0] = pair.have[0];
          status.have[1] = pair.have[1];
          status.skew_ns = pair.skew_ns;
          status.paused = paused;
          view->present(image[0], image[1], status);
        }
#endif
      }

      const auto now = std::chrono::steady_clock::now();
      if (now - last_report >= std::chrono::seconds(2)) {
        const double elapsed_s =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - last_report).count();
        // Frames leaving the pairer per second, paired or not -- deliberately
        // not called pairs/s, which read as 7.5 next to paired=0 and made an
        // unsynced rig look like a working one.
        const double rate_hz = static_cast<double>(emitted - last_report_emitted) / elapsed_s;
        last_report = now;
        last_report_emitted = emitted;

        std::string line = pairer.health_line();
        char rate[48];
        std::snprintf(rate, sizeof(rate), " out/s=%.1f", rate_hz);
        line += rate;
        if (live) {
          line += " | " + live->health_line();
          // Re-read per report, not per frame: this is a live GVCP register
          // read, and PTP state is the thing most likely to change under you
          // mid-run (BMC re-election, a master appearing or going away).
          // ptp_sample() carries OffsetFromMaster too, so the servo converging
          // -- or a camera re-link throwing it off -- is visible in the line.
          std::string ptp;
          for (const perception::LiveStereo::PtpSample& sample : live->ptp_sample()) {
            if (!ptp.empty()) ptp += "/";
            ptp += sample.status.empty() ? "-" : sample.status;
            if (sample.has_offset && sample.status == "Slave") {
              char off[24];
              std::snprintf(off, sizeof(off), "(%+.1fus)", sample.offset_ns / 1000.0);
              ptp += off;
            }
          }
          line += " | ptp " + ptp;
        }
        if (options.capture_hz > 0.0) {
          char trig[64];
          std::snprintf(trig, sizeof(trig), " | trig sent=%lu fail=%lu",
                        static_cast<unsigned long>(trig_sent.load()),
                        static_cast<unsigned long>(trig_failed.load()));
          line += trig;
        }
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
      }
    }

    // Stop the trigger loop before tearing down acquisition -- send_trigger
    // touches the SDK the sources own.
    trig_run.store(false, std::memory_order_relaxed);
    if (trigger_thread.joinable()) trigger_thread.join();

    if (live) live->stop();

    std::printf("\n%s\n", pairer.health_line().c_str());
    if (live) std::printf("%s\n", live->health_line().c_str());

    // How far apart the two eyes *claim* to have started, from the one instant
    // both were given. Worth printing precisely because it is a claim: it is
    // computed from the cameras' own clocks, so a shared clock error cancels
    // out of it exactly and leaves this number looking perfect.
    if (sync_check[0].have_first() && sync_check[1].have_first()) {
      const double delta_ms =
          static_cast<double>(static_cast<int64_t>(sync_check[1].first_timestamp_ns()) -
                              static_cast<int64_t>(sync_check[0].first_timestamp_ns())) *
          1e-6;
      std::printf("action_sync: first frames %.3f ms apart by the cameras' own clocks\n",
                  delta_ms);
    }

    if (pairer.paired() > 0) {
      std::printf(
          "\nNOTE: paired= and max_skew= are computed from the camera timestamps, so they\n"
          "measure agreement between the two clocks, not agreement with reality. A PTP\n"
          "offset error that is constant -- which is what an asymmetric or saturated link\n"
          "produces -- shifts a camera's exposures without shifting its timestamps, so it\n"
          "cancels out of every number above and reads as a perfectly synced rig.\n"
          "To actually verify sync, point both cameras at something whose timing does not\n"
          "come from either of them: a marked fan at a known rpm (3000 rpm = 18 deg/ms) or\n"
          "an LED strobed by a signal generator. Use a short exposure -- at 8 ms any offset\n"
          "smaller than the exposure is smeared into the same blur in both eyes.\n");
    }

    // A run that paired nothing is the failure this tool exists to catch, so it
    // is an exit code and not just a line in the log.
    if (pairer.paired() == 0) {
      // "No frames at all" and "frames that would not pair" are different
      // faults with different fixes, and saying "check ptp" for the first one
      // sends you to look at a clock that was working. emitted counts what left
      // the pairer, so zero means nothing was ever pushed into it.
      if (emitted == 0) {
        std::printf(
            "\nNO FRAMES: neither camera delivered anything, so there was nothing to pair.\n");
        if (action_sync_config.enabled) {
          std::printf(
              "A scheduled start was armed, which means the cameras are sitting on\n"
              "TriggerMode=On waiting for an Action Command. If the acks above did not all\n"
              "read OK, the command was never accepted and they will wait forever: check\n"
              "that ActionDeviceKey/ActionGroupKey/ActionGroupMask in camera.features match\n"
              "the action_sync section, and that the cameras are on the same subnet as the\n"
              "broadcast.\n");
        } else {
          std::printf(
              "Nothing was triggering here, so this is acquisition itself: check the camera\n"
              "is streaming (spin_acquire), the exposure and frame rate are sane, and the\n"
              "timeouts count above.\n");
        }
        return 1;
      }

      std::printf(
          "\nNO PAIRS: both cameras delivered frames, but none landed within %luus of each\n"
          "other. Either the clocks share no epoch (check ptp above -- Master/Master means\n"
          "they do not), or they do share one and the cameras are simply exposing at\n"
          "different times, which a scheduled start fixes and a wider tolerance only\n"
          "hides. --sync-start, or --capture-hz for every frame, is the way to tell.\n",
          static_cast<unsigned long>(config.tolerance_ns / 1000));
      return 1;
    }
  } catch (const std::exception& e) {
    std::printf("FAILED: %s\n", e.what());
    return 1;
  }

  return 0;
}
