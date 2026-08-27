#include "stereo_live.hpp"

#include <Spinnaker.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

#include "action_sync.hpp"
#include "heap_frame_sink.hpp"
#include "spinnaker_source.hpp"

namespace perception {
namespace {

// How long a reader parks in pop() before looking at the stop flag again.
constexpr std::chrono::milliseconds kPopTimeout{100};

}  // namespace

struct LiveStereo::Impl {
  struct Stream {
    std::unique_ptr<SpinnakerSource> source;
    std::unique_ptr<HeapFrameSink> sink;
    std::thread reader;
  };

  Spinnaker::SystemPtr system;
  Spinnaker::CameraList cameras;
  Stream streams[2];
  CameraGeometry geometry;
  FrameCallback callback;
  std::atomic<bool> running{false};
  // Separate from `running`, because stop() clears that one only partway
  // through and still has to be idempotent (the destructor calls it too).
  std::atomic<bool> stopping{false};

  // One reader per camera, rather than one thread polling both: the two sinks
  // are independent and a frame on one must not wait behind a frame on the
  // other, which is the whole reason the pairer takes per-stream locks.
  void read_loop(uint32_t id) {
    HeapFrameSink& sink = *streams[id].sink;
    while (running.load(std::memory_order_relaxed)) {
      HeapFrameSink::Frame frame;
      if (!sink.pop(frame, kPopTimeout)) continue;

      try {
        callback(id, frame.meta, frame.data);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "stereo: consumer threw on cam%u: %s\n", id, e.what());
      }

      // Released immediately, whatever the consumer did with it. Holding the
      // slot for the length of a viewer's redraw or a disk write is how a
      // recorder turns into camera backpressure; the consumers copy out
      // instead. See recording_plan.md, "Backpressure".
      sink.release(frame.slot);
    }
  }
};

LiveStereo::LiveStereo(const StereoCameras& config, const CameraConfig& camera,
                       FrameCallback callback)
    : impl_(new Impl) {
  impl_->callback = std::move(callback);
  impl_->system = Spinnaker::System::GetInstance();
  impl_->cameras = impl_->system->GetCameras();

  if (impl_->cameras.GetSize() < 2) {
    const unsigned int found = impl_->cameras.GetSize();
    impl_->cameras.Clear();
    impl_->system->ReleaseInstance();
    throw std::runtime_error("stereo: needs two cameras, found " + std::to_string(found));
  }

  for (uint32_t id = 0; id < 2; ++id) {
    CameraConfig per_camera = camera;
    per_camera.serial = config.serials[id];

    impl_->streams[id].source.reset(new SpinnakerSource(
        SpinnakerSource::select(impl_->cameras, per_camera.serial), per_camera));

    const CameraGeometry& geometry = impl_->streams[id].source->geometry();
    if (id == 0) {
      impl_->geometry = geometry;
    } else if (geometry.width != impl_->geometry.width ||
               geometry.height != impl_->geometry.height ||
               geometry.pixel_format != impl_->geometry.pixel_format ||
               geometry.frame_bytes != impl_->geometry.frame_bytes) {
      throw std::runtime_error(
          "stereo: the two cameras disagree about geometry (" + std::to_string(geometry.width) +
          "x" + std::to_string(geometry.height) + " " + geometry.pixel_format + " vs " +
          std::to_string(impl_->geometry.width) + "x" + std::to_string(impl_->geometry.height) +
          " " + impl_->geometry.pixel_format +
          ") -- both halves of a pair must be cut the same way");
    }

    if (config.buffer_count < impl_->streams[id].source->min_slot_count()) {
      throw std::runtime_error("stereo: buffer_count is below what this stream mode needs (" +
                               std::to_string(impl_->streams[id].source->min_slot_count()) + ")");
    }
    impl_->streams[id].sink.reset(
        new HeapFrameSink(config.buffer_count, impl_->geometry.buffer_bytes));
  }

  // Empty serials plus two cameras means select() returned the same camera
  // twice, and the run would look like one camera pairing with itself.
  if (config.serials[0] == config.serials[1]) {
    throw std::runtime_error(
        "stereo: both streams name the same serial (" +
        (config.serials[0].empty() ? std::string("unset") : config.serials[0]) +
        ") -- set streams[].serial so left and right cannot swap between runs");
  }
}

LiveStereo::~LiveStereo() {
  stop();
  if (impl_) {
    impl_->streams[0].source.reset();
    impl_->streams[1].source.reset();
    impl_->cameras.Clear();
    if (impl_->system) impl_->system->ReleaseInstance();
  }
}

void LiveStereo::start() {
  if (impl_->running.exchange(true)) return;
  for (uint32_t id = 0; id < 2; ++id) {
    impl_->streams[id].reader = std::thread(&Impl::read_loop, impl_.get(), id);
    impl_->streams[id].source->start(*impl_->streams[id].sink);
  }
}

void LiveStereo::stop() {
  if (!impl_ || !impl_->running.load(std::memory_order_relaxed)) return;
  if (impl_->stopping.exchange(true)) return;

  // Order matters, and getting it wrong is not subtle.
  //
  // SpinnakerSource::stop() tears the stream down and then waits for every
  // buffer the reader still holds to come back -- and the reader thread is the
  // only thing that hands them back. Clearing `running` first (which is what
  // this used to do) strands whatever was committed but not yet popped: the
  // drain times out after two seconds per camera, the camera fills its
  // remaining buffers and reports "input image buffer pool is exhausted", and
  // the source treats that as a stream error and starts reconnecting -- all
  // while shutting down.
  //
  // So: stop acquisition first, with the readers still draining behind it.
  for (Impl::Stream& stream : impl_->streams) {
    if (stream.source) stream.source->stop();
  }

  // Nothing can be committed any more, so the readers have nothing left to do.
  impl_->running.store(false, std::memory_order_relaxed);
  for (Impl::Stream& stream : impl_->streams) {
    if (stream.sink) stream.sink->stop();
    if (stream.reader.joinable()) stream.reader.join();
  }
}

const CameraGeometry& LiveStereo::geometry() const { return impl_->geometry; }

std::vector<std::string> LiveStereo::ptp_status() {
  return {impl_->streams[0].source->ptp_status(), impl_->streams[1].source->ptp_status()};
}

std::vector<LiveStereo::PtpSample> LiveStereo::ptp_sample() {
  std::vector<PtpSample> out;
  for (Impl::Stream& stream : impl_->streams) {
    PtpSample sample;
    sample.status = stream.source->ptp_status();
    sample.has_offset = stream.source->ptp_offset_ns(sample.offset_ns);
    out.push_back(std::move(sample));
  }
  return out;
}

bool LiveStereo::wait_for_ptp_slave(std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::string last;
  bool announced = false;

  for (;;) {
    const std::vector<std::string> statuses = ptp_status();
    if (statuses.size() == 2 && statuses[0] == "Slave" && statuses[1] == "Slave") {
      if (announced) std::printf("ptp: locked\n");
      return true;
    }

    std::string joined;
    for (const std::string& status : statuses) {
      if (!joined.empty()) joined += "/";
      joined += status.empty() ? "unsupported" : status;
    }
    if (joined != last) {
      // Only on change: polling this at 4 Hz would otherwise print forty
      // identical lines while nothing happens.
      std::printf("ptp: %s%s\n", joined.c_str(),
                  announced ? "" : " -- waiting for both cameras to reach Slave");
      last = joined;
      announced = true;
    }

    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
}

void LiveStereo::arm_scheduled_start(const ActionSyncConfig& config, ActionSyncChecker& left,
                                     ActionSyncChecker& right) {
  // Applying the camera config resets PTP, so both cameras are in Listening
  // right now and need a few Announce intervals. Waiting here rather than
  // sampling once is the difference between a rig that works and one that
  // reports NO_REF_TIME on every start.
  if (!wait_for_ptp_slave(std::chrono::milliseconds(config.ptp_wait_ms))) {
    // Falls through to arm_action_sync, which throws with the full explanation
    // of what the current status means. Nothing to add here beyond the fact
    // that waiting did not help.
    std::printf("ptp: still not locked after %ums\n", config.ptp_wait_ms);
  }

  // Check the keys before sending, not after. A camera whose keys do not match
  // the broadcast does not report an error -- it just does not ack, and with
  // TriggerMode On it then waits forever for a trigger it will never accept.
  // The visible symptom is "no acks received" plus delivered=0, which says
  // nothing about which of the three keys was wrong.
  for (uint32_t id = 0; id < 2; ++id) {
    uint32_t device_key = 0, group_key = 0, group_mask = 0;
    switch (impl_->streams[id].source->action_keys(device_key, group_key, group_mask)) {
      case SpinnakerSource::ActionKeys::Absent:
        throw std::runtime_error(
            "action_sync: cam" + std::to_string(id) +
            " has no ActionDeviceKey/ActionGroupKey/ActionGroupMask nodes, so it cannot take "
            "Action Commands at all -- a scheduled start is not available on this model");

      case SpinnakerSource::ActionKeys::WriteOnly:
        // Nothing to check against: the keys can be set but not read back. Say
        // so once, so that a later "no acks received" is read as "the keys the
        // config set do not match what the command sends" rather than as a
        // network problem.
        if (id == 0) {
          std::printf(
              "action_sync: the cameras' action keys are write-only, so they cannot be "
              "verified here.\n"
              "            If the acks below are not all OK, check that ActionDeviceKey, "
              "ActionGroupKey\n"
              "            and ActionGroupMask in camera.features match device_key=%u "
              "group_key=%u group_mask=%u.\n",
              config.device_key, config.group_key, config.group_mask);
        }
        break;

      case SpinnakerSource::ActionKeys::Readable:
        if (device_key == config.device_key && group_key == config.group_key &&
            group_mask == config.group_mask) {
          break;
        }
        throw std::runtime_error(
            "action_sync: cam" + std::to_string(id) + " has device_key=" +
            std::to_string(device_key) + " group_key=" + std::to_string(group_key) +
            " group_mask=" + std::to_string(group_mask) +
            ", but the command would be sent with " + std::to_string(config.device_key) + "/" +
            std::to_string(config.group_key) + "/" + std::to_string(config.group_mask) +
            ". A camera ignores a command whose keys do not match, silently. Set "
            "ActionDeviceKey, ActionGroupKey and ActionGroupMask in camera.features so they "
            "agree with the action_sync section.");
    }
  }

  // One broadcast, both cameras. That is the entire point of a *scheduled*
  // command over an unscheduled one: the instant is carried in the packet and
  // each camera's own firmware fires on its own PTP clock, so how long the
  // packet took to arrive -- and how differently it took to reach each camera
  // -- does not enter into it.
  arm_action_sync(impl_->system, ptp_status(), config, left);

  // The second checker is armed from the first: same scheduled instant, same
  // expectations. Only the label differs, so the two verdicts are tellable
  // apart and their first_timestamp_ns() can be compared against each other.
  right = left;
  left.label = "cam0";
  right.label = "cam1";
}

bool LiveStereo::send_trigger(const ActionSyncConfig& config, uint64_t target_ns) {
  // pResultSize is the EXPECTED device count -- two cameras. Passing the real
  // count (not an array capacity) is what keeps the call from waiting out the
  // ack timeout for devices that do not exist.
  Spinnaker::ActionCommandResult results[2];
  unsigned int result_size = 2;
  impl_->system->SendActionCommand(config.device_key, config.group_key, config.group_mask,
                                   target_ns, /*requestAck=*/true, &result_size, results);
  unsigned int ok = 0;
  for (unsigned int i = 0; i < result_size; ++i) {
    if (results[i].Status == Spinnaker::SPINNAKER_ACTION_COMMAND_STATUS_OK) ++ok;
  }
  return ok >= 2;
}

bool LiveStereo::failed() const {
  return impl_->streams[0].source->failed() || impl_->streams[1].source->failed();
}

std::string LiveStereo::health_line() const {
  std::string out;
  for (uint32_t id = 0; id < 2; ++id) {
    const SpinnakerSource& source = *impl_->streams[id].source;
    char line[192];
    std::snprintf(line, sizeof(line), "%scam%u delivered=%lu incomplete=%lu timeouts=%lu",
                  id ? " | " : "", id, static_cast<unsigned long>(source.delivered()),
                  static_cast<unsigned long>(source.incomplete()),
                  static_cast<unsigned long>(source.timeouts()));
    out += line;
  }
  return out;
}

}  // namespace perception
