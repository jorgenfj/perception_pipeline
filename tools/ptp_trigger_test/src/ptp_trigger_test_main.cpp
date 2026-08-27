// Headless PTP scheduled-trigger drop / speed test.
//
//   ptp_trigger_test [options] [config.yaml]
//
// Puts the camera(s) in per-frame FrameStart / Action0 trigger mode, then fires
// one *scheduled* Action Command per frame and checks two independent things:
//
//   1. Command loss on the wire -- every SendActionCommand asks for an ack, so
//      a camera that does not answer means the command did not land there. A
//      non-OK ack (NO_REF_TIME / OVERFLOW / ACTION_LATE / ERROR) means it
//      landed and was refused, or the ack lost the race with image traffic on
//      the return path.
//   2. Frame loss after a trigger -- each command is one trigger, so each
//      camera should return one frame. delivered < sent means the trigger
//      fired but the image never made it back (bandwidth / link loss).
//
// Cameras:
//   default        both cameras from the config's streams (a synced stereo run)
//   --camera SER   just that serial -- e.g. speed-test the gigabit camera on its
//                  own, so a starved second camera does not muddy the table.
//
// Modes:
//   default        hold one rate (--hz) and print a full drop report.
//   --sweep a,b,c  step through rates and print an fps/drop table -- the way to
//                  read a link's clean synced-capture ceiling.
//
// No visualization, no recording: it prints counters and a verdict.

#include <Spinnaker.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "camera_config.hpp"
#include "config_path.hpp"
#include "frame_sink.hpp"  // host_now_ns(), FrameMeta
#include "heap_frame_sink.hpp"
#include "spinnaker_source.hpp"
#include "stereo_config.hpp"

#ifndef PERCEPTION_CONFIG_DIR
#define PERCEPTION_CONFIG_DIR ""
#endif

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

constexpr uint32_t kMaxCameras = 2;

const char* status_name(Spinnaker::ActionCommandStatus status) {
  switch (status) {
    case Spinnaker::SPINNAKER_ACTION_COMMAND_STATUS_OK:
      return "OK";
    case Spinnaker::SPINNAKER_ACTION_COMMAND_STATUS_NO_REF_TIME:
      return "NO_REF_TIME";
    case Spinnaker::SPINNAKER_ACTION_COMMAND_STATUS_OVERFLOW:
      return "OVERFLOW";
    case Spinnaker::SPINNAKER_ACTION_COMMAND_STATUS_ACTION_LATE:
      return "ACTION_LATE";
    case Spinnaker::SPINNAKER_ACTION_COMMAND_STATUS_ERROR:
      return "ERROR";
    default:
      return "unknown";
  }
}

std::string ip_str(uint32_t addr) {
  // DeviceAddress comes back host-order on this little-endian box, so the low
  // byte is the last octet -- print low-to-high to get the real dotted quad.
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%u.%u.%u.%u", addr & 0xFFu, (addr >> 8) & 0xFFu,
                (addr >> 16) & 0xFFu, (addr >> 24) & 0xFFu);
  return buf;
}

// One-to-N cameras straight off SpinnakerSource, so this tool can run a single
// camera -- which LiveStereo, fixed at two, cannot. Same open/config/PTP/drain
// as LiveStereo, generalized over count.
class Rig {
 public:
  using FrameCb = std::function<void(uint32_t stream, const perception::FrameMeta&, const void*)>;

  Rig(const std::vector<std::string>& serials, const perception::CameraConfig& base,
      uint32_t buffer_count, FrameCb cb)
      : cb_(std::move(cb)) {
    system_ = Spinnaker::System::GetInstance();
    cameras_ = system_->GetCameras();
    if (cameras_.GetSize() < serials.size()) {
      const unsigned int found = cameras_.GetSize();
      cameras_.Clear();
      system_->ReleaseInstance();
      throw std::runtime_error("need " + std::to_string(serials.size()) + " camera(s), found " +
                               std::to_string(found));
    }
    streams_.resize(serials.size());
    for (std::size_t id = 0; id < serials.size(); ++id) {
      perception::CameraConfig cfg = base;
      cfg.serial = serials[id];
      streams_[id].src.reset(new perception::SpinnakerSource(
          perception::SpinnakerSource::select(cameras_, cfg.serial), cfg));
      const perception::CameraGeometry& g = streams_[id].src->geometry();
      if (buffer_count < streams_[id].src->min_slot_count()) {
        throw std::runtime_error("buffer_count below what this stream mode needs (" +
                                 std::to_string(streams_[id].src->min_slot_count()) + ")");
      }
      streams_[id].sink.reset(new perception::HeapFrameSink(buffer_count, g.buffer_bytes));
    }
  }

  ~Rig() {
    stop();
    streams_.clear();
    cameras_.Clear();
    if (system_) system_->ReleaseInstance();
  }

  Rig(const Rig&) = delete;
  Rig& operator=(const Rig&) = delete;

  uint32_t count() const { return static_cast<uint32_t>(streams_.size()); }
  Spinnaker::SystemPtr& system() { return system_; }

  void start() {
    if (running_.exchange(true)) return;
    for (uint32_t id = 0; id < count(); ++id) {
      streams_[id].reader = std::thread(&Rig::read_loop, this, id);
      streams_[id].src->start(*streams_[id].sink);
    }
  }

  void stop() {
    if (!running_.load(std::memory_order_relaxed)) return;
    if (stopping_.exchange(true)) return;
    for (auto& s : streams_)
      if (s.src) s.src->stop();  // acquisition down first, readers still draining
    running_.store(false, std::memory_order_relaxed);
    for (auto& s : streams_) {
      if (s.sink) s.sink->stop();
      if (s.reader.joinable()) s.reader.join();
    }
  }

  std::vector<std::string> ptp_status() {
    std::vector<std::string> out;
    for (auto& s : streams_) out.push_back(s.src->ptp_status());
    return out;
  }

  bool wait_for_ptp_slave(std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::string last;
    bool announced = false;
    for (;;) {
      const std::vector<std::string> st = ptp_status();
      bool all = !st.empty();
      std::string joined;
      for (const std::string& s : st) {
        if (s != "Slave") all = false;
        if (!joined.empty()) joined += "/";
        joined += s.empty() ? "unsupported" : s;
      }
      if (all) {
        if (announced) std::printf("ptp: locked\n");
        return true;
      }
      if (joined != last) {
        std::printf("ptp: %s%s\n", joined.c_str(), announced ? "" : " -- waiting for Slave");
        last = joined;
        announced = true;
      }
      if (std::chrono::steady_clock::now() >= deadline) return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
  }

  bool failed() const {
    for (const auto& s : streams_)
      if (s.src->failed()) return true;
    return false;
  }

  std::string health_line() const {
    std::string out;
    for (uint32_t id = 0; id < count(); ++id) {
      const perception::SpinnakerSource& src = *streams_[id].src;
      char line[192];
      std::snprintf(line, sizeof(line), "%scam%u delivered=%lu incomplete=%lu timeouts=%lu",
                    id ? " | " : "", id, static_cast<unsigned long>(src.delivered()),
                    static_cast<unsigned long>(src.incomplete()),
                    static_cast<unsigned long>(src.timeouts()));
      out += line;
    }
    return out;
  }

 private:
  struct Stream {
    std::unique_ptr<perception::SpinnakerSource> src;
    std::unique_ptr<perception::HeapFrameSink> sink;
    std::thread reader;
  };

  void read_loop(uint32_t id) {
    perception::HeapFrameSink& sink = *streams_[id].sink;
    while (running_.load(std::memory_order_relaxed)) {
      perception::HeapFrameSink::Frame frame;
      if (!sink.pop(frame, std::chrono::milliseconds(100))) continue;
      try {
        cb_(id, frame.meta, frame.data);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "cam%u consumer threw: %s\n", id, e.what());
      }
      sink.release(frame.slot);
    }
  }

  Spinnaker::SystemPtr system_;
  Spinnaker::CameraList cameras_;
  std::vector<Stream> streams_;
  FrameCb cb_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stopping_{false};
};

struct Options {
  std::string config_path;
  std::string camera;      // one serial => single-camera run; empty => both from config
  double hz = 5.0;
  uint64_t frames = 200;   // 0 = until Ctrl-C
  double lead_ms = 100.0;
  uint32_t ptp_wait_ms = 20000;
  std::vector<double> sweep;
  double secs = 4.0;
  bool request_ack = true;  // --no-ack: fire-and-forget, so the sender is not
                            // throttled by the blocking ack wait (speed test)
};

void print_usage() {
  std::printf(
      "ptp_trigger_test [options] [config.yaml]\n"
      "\n"
      "  --camera SER   test just this serial (single camera), instead of both\n"
      "  --hz N         scheduled captures per second (default 5)\n"
      "  --frames N     stop after N commands, 0 runs until Ctrl-C (default 200)\n"
      "  --lead-ms N    schedule each command N ms ahead of its fire time (default 100)\n"
      "  --ptp-wait N   ms to wait for the camera(s) to reach PTP Slave (default 20000)\n"
      "  --sweep a,b,c  step through these rates and print an fps/drop table\n"
      "  --secs N       seconds to hold each --sweep rate (default 4)\n"
      "  --no-ack       fire-and-forget: don't wait for acks. Use for a speed test --\n"
      "                 the blocking ack wait otherwise caps the send rate to the ack\n"
      "                 round-trip (~8 Hz on a slow link), not the camera's throughput.\n");
}

std::vector<double> parse_rates(const std::string& csv) {
  std::vector<double> out;
  std::size_t i = 0;
  while (i < csv.size()) {
    std::size_t comma = csv.find(',', i);
    if (comma == std::string::npos) comma = csv.size();
    const std::string tok = csv.substr(i, comma - i);
    if (!tok.empty()) {
      const double hz = std::stod(tok);
      if (hz <= 0.0) throw std::runtime_error("--sweep rates must be positive");
      out.push_back(hz);
    }
    i = comma + 1;
  }
  if (out.empty()) throw std::runtime_error("--sweep needs at least one rate, e.g. 10,20,40");
  return out;
}

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string(what) + " needs a value");
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      print_usage();
      std::exit(0);
    } else if (a == "--camera") {
      o.camera = next("--camera");
    } else if (a == "--hz") {
      o.hz = std::stod(next("--hz"));
    } else if (a == "--frames") {
      o.frames = std::stoull(next("--frames"));
    } else if (a == "--lead-ms") {
      o.lead_ms = std::stod(next("--lead-ms"));
    } else if (a == "--ptp-wait") {
      o.ptp_wait_ms = static_cast<uint32_t>(std::stoul(next("--ptp-wait")));
    } else if (a == "--sweep") {
      o.sweep = parse_rates(next("--sweep"));
    } else if (a == "--secs") {
      o.secs = std::stod(next("--secs"));
    } else if (a == "--no-ack") {
      o.request_ack = false;
    } else if (!a.empty() && a[0] == '-') {
      throw std::runtime_error("unknown option '" + a + "'");
    } else if (o.config_path.empty()) {
      o.config_path = a;
    } else {
      throw std::runtime_error("more than one config file given ('" + a + "')");
    }
  }
  if (o.hz <= 0.0) throw std::runtime_error("--hz must be positive");
  return o;
}

// The camera-side setup a per-frame scheduled trigger needs, appended after the
// config's own features so it wins. Order matters: keys and selector/source
// before TriggerMode On, and the frame-rate limiter off so the trigger paces
// acquisition.
void append_trigger_features(perception::CameraConfig& camera,
                             const perception::ActionSyncConfig& sync) {
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
  // writable when the source is Action0 (there is no edge on an action).
  set("TriggerMode", "On");
}

struct BurstStats {
  uint64_t sent = 0;
  uint64_t acks_total = 0;
  uint64_t all_ok = 0;       // commands where >= ncam cameras acked OK
  uint64_t missing_ack = 0;  // fewer acks than cameras: lost on the wire
  std::map<int, uint64_t> status_counts;
  std::map<uint32_t, uint64_t> ok_by_ip;
  uint64_t delivered[kMaxCameras] = {0, 0};
  double send_secs = 0.0;
  uint64_t send_us_sum = 0;  // time spent inside SendActionCommand
  uint64_t send_us_max = 0;

  double send_ms_mean() const { return sent ? (send_us_sum / static_cast<double>(sent)) / 1e3 : 0.0; }
  uint64_t ok_acks() const {
    auto it = status_counts.find(Spinnaker::SPINNAKER_ACTION_COMMAND_STATUS_OK);
    return it == status_counts.end() ? 0 : it->second;
  }
  double fps(uint32_t cam) const { return send_secs > 0 ? delivered[cam] / send_secs : 0.0; }
  uint64_t dropped(uint32_t cam) const {
    return sent > delivered[cam] ? sent - delivered[cam] : 0;
  }
};

BurstStats run_burst(Spinnaker::SystemPtr& system, const perception::ActionSyncConfig& sync,
                     double hz, uint64_t frames, uint64_t lead_ns, Rig& rig,
                     std::atomic<uint64_t> frames_seen[kMaxCameras], uint32_t ncam, bool request_ack,
                     bool verbose) {
  BurstStats st;
  const uint64_t period_ns = static_cast<uint64_t>(1e9 / hz);

  uint64_t start_seen[kMaxCameras] = {0, 0};
  for (uint32_t s = 0; s < ncam; ++s) start_seen[s] = frames_seen[s].load();

  const uint64_t started_ns = perception::host_now_ns();
  uint64_t target = started_ns + lead_ns;
  uint64_t next_log = started_ns + 1'000'000'000ull;

  while (!g_stop && (frames == 0 || st.sent < frames)) {
    Spinnaker::ActionCommandResult results[8];
    // pResultSize is the EXPECTED device count, and the call blocks until that
    // many acks arrive OR an internal timeout fires. It must be the real camera
    // count -- passing the array capacity (8) made every call wait the full
    // timeout for 6-7 devices that do not exist (~126 ms, the whole rate cap).
    // NULL => return as soon as the command is broadcast: fire-and-forget, the
    // fast path for a throughput test.
    unsigned int result_size = ncam;
    const auto call0 = std::chrono::steady_clock::now();
    if (request_ack) {
      system->SendActionCommand(sync.device_key, sync.group_key, sync.group_mask, target,
                                /*requestAck=*/true, &result_size, results);
    } else {
      system->SendActionCommand(sync.device_key, sync.group_key, sync.group_mask, target,
                                /*requestAck=*/false, nullptr, nullptr);
      result_size = 0;
    }
    const uint64_t call_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - call0)
            .count());
    ++st.sent;
    st.send_us_sum += call_us;
    if (call_us > st.send_us_max) st.send_us_max = call_us;

    if (request_ack) {
      st.acks_total += result_size;
      unsigned int ok = 0;
      for (unsigned int i = 0; i < result_size; ++i) {
        ++st.status_counts[results[i].Status];
        if (results[i].Status == Spinnaker::SPINNAKER_ACTION_COMMAND_STATUS_OK) {
          ++st.ok_by_ip[results[i].DeviceAddress];
          ++ok;
        }
      }
      // >= ncam rather than == ncam: an idle second camera left in trigger mode
      // from a prior run may also ack, and that must not read as a failure.
      if (result_size < ncam) ++st.missing_ack;
      if (ok >= ncam) ++st.all_ok;
    }

    const uint64_t now = perception::host_now_ns();
    if (verbose && now >= next_log) {
      std::string frs;
      for (uint32_t s = 0; s < ncam; ++s) {
        frs += " cam" + std::to_string(s) + "=" +
               std::to_string(frames_seen[s].load() - start_seen[s]);
      }
      std::printf("t=%.1fs sent=%lu ok=%lu missing_ack=%lu | frames%s\n",
                  (now - started_ns) * 1e-9, static_cast<unsigned long>(st.sent),
                  static_cast<unsigned long>(st.all_ok),
                  static_cast<unsigned long>(st.missing_ack), frs.c_str());
      next_log += 1'000'000'000ull;
    }
    if (rig.failed()) break;

    target += period_ns;
    const int64_t wake = static_cast<int64_t>(target) - static_cast<int64_t>(lead_ns);
    const int64_t delta = wake - static_cast<int64_t>(perception::host_now_ns());
    if (delta > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(delta));
  }

  st.send_secs = (perception::host_now_ns() - started_ns) * 1e-9;
  std::this_thread::sleep_for(std::chrono::milliseconds(500) + std::chrono::nanoseconds(lead_ns));
  for (uint32_t s = 0; s < ncam; ++s) st.delivered[s] = frames_seen[s].load() - start_seen[s];
  return st;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);

  try {
    const Options options = parse_args(argc, argv);
    const std::string config_path =
        options.config_path.empty()
            ? perception::resolve_config_path("stereo.yaml", PERCEPTION_CONFIG_DIR)
            : options.config_path;

    perception::StereoConfig stereo = perception::load_stereo_config(config_path);
    perception::CameraConfig camera = perception::load_camera_config(config_path);
    const perception::ActionSyncConfig sync = perception::load_action_sync_config(config_path);
    append_trigger_features(camera, sync);

    // Which cameras: one named serial, or both from the config's streams.
    std::vector<std::string> serials;
    if (!options.camera.empty()) {
      serials.push_back(options.camera);
    } else {
      for (uint32_t s = 0; s < kMaxCameras; ++s) serials.push_back(stereo.streams[s].serial);
    }
    const uint32_t ncam = static_cast<uint32_t>(serials.size());
    const uint64_t lead_ns = static_cast<uint64_t>(options.lead_ms * 1e6);

    std::printf("config: %s\n", config_path.c_str());
    std::printf("cameras:");
    for (uint32_t s = 0; s < ncam; ++s)
      std::printf(" cam%u=%s", s, serials[s].empty() ? "?" : serials[s].c_str());
    std::printf("\n");

    std::atomic<uint64_t> frames_seen[kMaxCameras] = {};
    Rig rig(serials, camera, stereo.buffer_count,
            [&](uint32_t stream, const perception::FrameMeta&, const void*) {
              if (stream < kMaxCameras) frames_seen[stream].fetch_add(1, std::memory_order_relaxed);
            });

    rig.start();  // BeginAcquisition: cameras armed, waiting for Action0 triggers
    std::printf("waiting for PTP lock (Slave)...\n");
    if (!rig.wait_for_ptp_slave(std::chrono::milliseconds(options.ptp_wait_ms))) {
      std::printf(
          "WARNING: not locked -- sending anyway so the test reports the NO_REF_TIME an "
          "unsynced camera returns.\n");
    }

    Spinnaker::SystemPtr system = rig.system();

    // ---- sweep mode ---------------------------------------------------------
    if (!options.sweep.empty()) {
      std::printf("sweep: %.1fs per rate, lead %.0f ms, acks %s\n\n", options.secs, options.lead_ms,
                  options.request_ack ? "on (send rate limited by ack round-trip)"
                                      : "off (fire-and-forget, true throughput)");
      std::printf(" req_hz |");
      for (uint32_t s = 0; s < ncam; ++s) std::printf(" cam%u_fps", s);
      std::printf(" |");
      for (uint32_t s = 0; s < ncam; ++s) std::printf(" drop%u", s);
      std::printf(" | send_ms | ack_ok%%\n");
      for (double hz : options.sweep) {
        if (g_stop) break;
        const uint64_t n = static_cast<uint64_t>(hz * options.secs + 0.5);
        const BurstStats st = run_burst(system, sync, hz, n ? n : 1, lead_ns, rig, frames_seen,
                                        ncam, options.request_ack, false);
        std::string flag;
        for (uint32_t s = 0; s < ncam; ++s) {
          if (st.dropped(s)) flag += (flag.empty() ? "  <- DROPS: cam" : " cam") +
                                     std::to_string(s) + "(" + serials[s] + ")";
        }
        std::printf("%7.1f |", hz);
        for (uint32_t s = 0; s < ncam; ++s) std::printf(" %8.1f", st.fps(s));
        std::printf(" |");
        for (uint32_t s = 0; s < ncam; ++s) std::printf(" %5lu", static_cast<unsigned long>(st.dropped(s)));
        std::printf(" | %7.1f | ", st.send_ms_mean());
        if (options.request_ack)
          std::printf("%6.0f%s\n", st.acks_total ? 100.0 * st.ok_acks() / st.acks_total : 0.0,
                      flag.c_str());
        else
          std::printf("     -%s\n", flag.c_str());
        std::fflush(stdout);
        if (rig.failed()) {
          std::printf("a camera gave up -- stopping sweep\n");
          break;
        }
      }
      rig.stop();
      std::printf(
          "\nRead the ceiling as the highest req_hz with all drop columns 0. Above it, a\n"
          "camera's real sustained rate is its cam*_fps -- that is the link's limit.\n"
          "send_ms is the mean SendActionCommand duration. With --no-ack it should be ~0;\n"
          "with acks on it is the time to collect them -- if it is ~1000/plateau_fps, the\n"
          "sender is the cap, not the link.\n");
      return 0;
    }

    // ---- single-rate mode ---------------------------------------------------
    std::printf("plan: %.1f Hz, %s, lead %.0f ms, keys %u/%u/%u\n", options.hz,
                options.frames ? (std::to_string(options.frames) + " commands").c_str()
                               : "until Ctrl-C",
                options.lead_ms, sync.device_key, sync.group_key, sync.group_mask);

    const BurstStats st = run_burst(system, sync, options.hz, options.frames, lead_ns, rig,
                                    frames_seen, ncam, options.request_ack, true);
    const std::string health = rig.health_line();
    rig.stop();

    std::printf("\n==== PTP scheduled-trigger drop test ====\n");
    std::printf("ran %.1fs at a requested %.1f Hz, %u camera(s)\n", st.send_secs, options.hz, ncam);
    std::printf("send-call duration   : mean %.1f ms, max %.1f ms  (%.1f Hz max send rate)%s\n",
                st.send_ms_mean(), st.send_us_max / 1e3,
                st.send_ms_mean() > 0 ? 1e3 / st.send_ms_mean() : 0.0,
                options.request_ack ? "" : "  [acks off]");

    if (options.request_ack) {
      std::printf("\n-- Action commands (wire level) --\n");
      std::printf("sent                 : %lu\n", static_cast<unsigned long>(st.sent));
      std::printf("acks received        : %lu  (expected %lu = %u per command)\n",
                  static_cast<unsigned long>(st.acks_total),
                  static_cast<unsigned long>(st.sent * ncam), ncam);
      std::printf("cameras acked OK     : %lu / %lu\n", static_cast<unsigned long>(st.all_ok),
                  static_cast<unsigned long>(st.sent));
      std::printf("missing an ack       : %lu   <-- command lost on the wire to a camera\n",
                  static_cast<unsigned long>(st.missing_ack));
      std::printf("ack status breakdown :");
      for (const auto& [status, count] : st.status_counts) {
        std::printf(" %s=%lu", status_name(static_cast<Spinnaker::ActionCommandStatus>(status)),
                    static_cast<unsigned long>(count));
      }
      std::printf("\nper-camera OK acks   :");
      for (const auto& [ip, count] : st.ok_by_ip) {
        std::printf(" %s=%lu", ip_str(ip).c_str(), static_cast<unsigned long>(count));
      }
      std::printf("\n");
    } else {
      std::printf("\n-- Action commands: acks off (--no-ack), so wire-level loss is not measured; "
                  "frame delivery below is the throughput signal --\n");
    }

    std::printf("\n-- Frames returned --\n%s\n", health.c_str());

    std::printf("\n-- Cross-check (each command is one trigger -> one frame per camera) --\n");
    uint64_t frames_dropped = 0;
    for (uint32_t s = 0; s < ncam; ++s) {
      std::printf("cam%u (serial %s): commands=%lu delivered=%lu dropped=%lu  (%.1f fps)\n", s,
                  serials[s].empty() ? "?" : serials[s].c_str(),
                  static_cast<unsigned long>(st.sent),
                  static_cast<unsigned long>(st.delivered[s]),
                  static_cast<unsigned long>(st.dropped(s)), st.fps(s));
      frames_dropped += st.dropped(s);
    }

    const uint64_t non_ok_acks = st.acks_total - st.ok_acks();
    std::printf("\nVERDICT: ");
    if (frames_dropped == 0 && st.missing_ack == 0 && non_ok_acks == 0) {
      std::printf("clean -- every command acked OK and every trigger returned a frame.\n");
      return 0;
    }
    if (frames_dropped > 0) {
      std::printf("frames dropped -- a camera could not service every trigger (bandwidth). "
                  "Lower --hz or shrink the frame; see the per-camera lines above.\n");
    }
    if (st.missing_ack > 0) {
      std::printf("         %lu command(s) went unacked -- lost on the wire.\n",
                  static_cast<unsigned long>(st.missing_ack));
    }
    if (non_ok_acks > 0 && frames_dropped == 0) {
      std::printf("         %lu ack(s) came back non-OK but frames still arrived -- ack-path "
                  "congestion, not lost captures. Raise --lead-ms.\n",
                  static_cast<unsigned long>(non_ok_acks));
    }
    return 1;
  } catch (const std::exception& e) {
    std::printf("FAILED: %s\n", e.what());
    return 1;
  }
}
