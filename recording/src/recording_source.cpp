#include "recording_source.hpp"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace perception {
namespace {

// Never sleep longer than this in one go, so stop() is acted on promptly even
// when the next frame is a long dropout away.
constexpr auto kMaxSleep = std::chrono::milliseconds(20);

}  // namespace

RecordingSource::RecordingSource(const Config& config) : config_(config) {
  if (config_.speed <= 0.0) throw std::runtime_error("recording source: speed must be positive");

  reader_ = std::make_unique<RecordingReader>(config_.directory);

  if (!config_.role.empty()) {
    bool found = false;
    for (std::size_t s = 0; s < reader_->stream_count(); ++s) {
      if (reader_->stream(s).role == config_.role) {
        stream_ = static_cast<uint32_t>(s);
        found = true;
        break;
      }
    }
    if (!found) {
      std::string roles;
      for (std::size_t s = 0; s < reader_->stream_count(); ++s) {
        if (!roles.empty()) roles += ", ";
        roles += reader_->stream(s).role;
      }
      throw std::runtime_error("recording source: no stream with role '" + config_.role +
                               "' (have: " + roles + ")");
    }
  } else {
    if (config_.stream >= reader_->stream_count()) {
      throw std::runtime_error("recording source: stream " + std::to_string(config_.stream) +
                               " but the recording has " +
                               std::to_string(reader_->stream_count()));
    }
    stream_ = config_.stream;
  }

  const StreamInfo& info = reader_->stream(stream_);
  if (reader_->index(stream_).empty()) {
    throw std::runtime_error("recording source: stream " + std::to_string(stream_) +
                             " (" + info.role + ") has no frames");
  }

  geometry_.width = info.width;
  geometry_.height = info.height;
  geometry_.stride_bytes = info.stride_bytes;
  geometry_.pixel_format = info.pixel_format;
  geometry_.frame_bytes = info.frame_bytes;
  // No transport padding to allow for: nothing is DMAing into these slots. The
  // camera path's buffer_bytes is larger only because a USB3 packet has to fit.
  geometry_.buffer_bytes = info.frame_bytes;
}

RecordingSource::~RecordingSource() { stop(); }

void RecordingSource::start(FrameSink& sink) {
  if (running_.exchange(true)) return;

  if (sink.slot_bytes() < geometry_.buffer_bytes) {
    running_.store(false);
    throw std::runtime_error("RecordingSource: sink slots are smaller than frame_bytes");
  }
  if (sink.slot_count() < min_slot_count()) {
    running_.store(false);
    throw std::runtime_error("RecordingSource: needs at least " +
                             std::to_string(min_slot_count()) + " slots; the sink has " +
                             std::to_string(sink.slot_count()));
  }

  held_.assign(sink.slot_count(), false);
  thread_ = std::thread(&RecordingSource::run, this, std::ref(sink));
}

void RecordingSource::stop() {
  running_.store(false, std::memory_order_relaxed);
  if (thread_.joinable()) thread_.join();
}

void RecordingSource::finish(std::string failure_reason) {
  if (finished_.load(std::memory_order_relaxed)) return;
  if (!failure_reason.empty()) {
    failure_ = std::move(failure_reason);
    failed_.store(true, std::memory_order_release);
  }
  finished_.store(true, std::memory_order_release);
  if (on_finished_) {
    try {
      on_finished_();
    } catch (...) {
    }
  }
}

void RecordingSource::reclaim(FrameSink& sink) {
  for (uint32_t slot = 0; slot < held_.size(); ++slot) {
    if (held_[slot] && sink.consumed(slot)) held_[slot] = false;
  }
}

uint32_t RecordingSource::acquire_slot(FrameSink& sink, bool& waited) {
  waited = false;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(config_.slot_wait_ms);
  for (;;) {
    reclaim(sink);
    for (uint32_t slot = 0; slot < held_.size(); ++slot) {
      if (!held_[slot]) return slot;
    }
    if (!running_.load(std::memory_order_relaxed)) return FrameSink::kNoSlot;
    if (std::chrono::steady_clock::now() >= deadline) return FrameSink::kNoSlot;
    waited = true;
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
}

void RecordingSource::run(FrameSink& sink) {
  const std::vector<IndexRecord>& index = reader_->index(stream_);
  const uint64_t epoch = reader_->manifest().epoch_ns;

  try {
    do {
      // One origin per pass, in both clocks: `origin` paces the wall-clock
      // wait, `base` is what rebased timestamps are measured from. Captured
      // together so the reported latency comes out at roughly zero rather than
      // at the gap between the two reads.
      const auto origin = std::chrono::steady_clock::now();
      const uint64_t base = host_now_ns();

      for (std::size_t i = 0; i < index.size(); ++i) {
        if (!running_.load(std::memory_order_relaxed)) return;

        const IndexRecord& record = index[i];

        // Pacing is from the frame's own offset into the recording, never from
        // "previous frame plus a nominal period", so a dropout replays as a
        // stall of exactly the right length.
        const uint64_t offset_ns = record.timestamp_ns >= epoch
                                       ? static_cast<uint64_t>(
                                             static_cast<double>(record.timestamp_ns - epoch) /
                                             config_.speed)
                                       : 0;
        const auto due = origin + std::chrono::nanoseconds(offset_ns);

        for (;;) {
          if (!running_.load(std::memory_order_relaxed)) return;
          const auto now = std::chrono::steady_clock::now();
          if (now >= due) break;
          std::this_thread::sleep_for(std::min(
              std::chrono::duration_cast<std::chrono::nanoseconds>(due - now),
              std::chrono::duration_cast<std::chrono::nanoseconds>(kMaxSleep)));
        }

        bool waited = false;
        const uint32_t slot = acquire_slot(sink, waited);
        if (waited) late_.fetch_add(1, std::memory_order_relaxed);
        if (slot == FrameSink::kNoSlot) {
          if (!running_.load(std::memory_order_relaxed)) return;
          slot_drops_.fetch_add(1, std::memory_order_relaxed);
          continue;
        }

        // Straight from the .dat into the sink's slot, no staging buffer.
        reader_->read_frame(stream_, i, sink.buffers()[slot]);

        FrameMeta meta;
        meta.timestamp_ns = config_.rebase_timestamps ? base + offset_ns : record.timestamp_ns;
        meta.host_recv_ns = host_now_ns();
        meta.frame_id = record.frame_id;
        meta.bytes = record.bytes;

        held_[slot] = true;
        sink.commit(slot, meta);
        delivered_.fetch_add(1, std::memory_order_relaxed);
      }

      if (config_.loop && running_.load(std::memory_order_relaxed)) {
        loops_.fetch_add(1, std::memory_order_relaxed);
      }
    } while (config_.loop && running_.load(std::memory_order_relaxed));
  } catch (const std::exception& e) {
    finish(e.what());
    return;
  }

  // Played to the end with looping off. Not a failure: the run is simply over,
  // and the owner is told so it stops waiting for a publish that is not coming.
  finish({});
}

std::string RecordingSource::counters() const {
  std::ostringstream out;
  out << "late=" << late() << " slot_drops=" << slot_drops() << " loops=" << loops();
  return out.str();
}

std::string RecordingSource::notes() const {
  if (slot_drops() == 0) return {};
  return "(pipeline slower than the recorded rate)";
}

std::string RecordingSource::ptp_status() {
  const std::string& recorded = reader_->manifest().ptp_status_at_start;
  return recorded.empty() ? std::string{} : "recorded:" + recorded;
}

}  // namespace perception
