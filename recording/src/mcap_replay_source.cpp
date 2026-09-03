#include "mcap_replay_source.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>

#include <mcap/mcap.hpp>

#include "cdr_reader.hpp"

namespace perception {
namespace {

// Never sleep longer than this in one go, so stop() is acted on promptly even
// when the next frame is a long dropout away.
constexpr auto kMaxSleep = std::chrono::milliseconds(20);

constexpr std::string_view kImageSchema = "sensor_msgs/msg/Image";

// The inverse of ros_messages.cpp's ros_encoding(), back into the names
// CameraGeometry::pixel_format uses -- the GenICam spelling, which is what
// to_image_desc() looks up. A file whose encoding is not here can still be
// listed and inspected; it just cannot be replayed through this pipeline.
const char* camera_format_of(const std::string& ros_encoding) {
  static const std::unordered_map<std::string, const char*> kFormats = {
      {"bayer_rggb8", "BayerRG8"}, {"bayer_grbg8", "BayerGR8"}, {"bayer_gbrg8", "BayerGB8"},
      {"bayer_bggr8", "BayerBG8"}, {"mono8", "Mono8"},          {"rgb8", "RGB8"},
      {"rgba8", "RGBa8"},
  };
  const auto it = kFormats.find(ros_encoding);
  return it == kFormats.end() ? nullptr : it->second;
}

// One sensor_msgs/Image, as far as a replay needs it. `pixels` points into the
// message, so it is valid only while that message is.
struct DecodedImage {
  uint64_t stamp_ns = 0;
  std::string frame_id;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t step = 0;
  std::string encoding;
  const unsigned char* pixels = nullptr;
  std::size_t bytes = 0;
};

bool decode_image(const std::byte* data, std::size_t size, DecodedImage& out) {
  CdrReader cdr(data, size);
  const int32_t sec = cdr.i32();
  const uint32_t nsec = cdr.u32();
  out.frame_id = cdr.str();
  out.height = cdr.u32();
  out.width = cdr.u32();
  out.encoding = cdr.str();
  cdr.u8();  // is_bigendian
  out.step = cdr.u32();
  if (!cdr.bytes(out.pixels, out.bytes)) return false;
  if (!cdr.ok() || sec < 0) return false;

  out.stamp_ns = static_cast<uint64_t>(sec) * 1'000'000'000ull + nsec;
  return true;
}

}  // namespace

// The reader has to outlive every message view taken from it, and mcap.hpp is
// not something the header should drag into every translation unit that only
// wants to construct one of these.
struct McapReplaySource::Impl {
  mcap::McapReader reader;
};

McapReplaySource::McapReplaySource(const Config& config)
    : config_(config), impl_(std::make_unique<Impl>()) {
  if (config_.speed <= 0.0) throw std::runtime_error("mcap replay: speed must be positive");

  const mcap::Status opened = impl_->reader.open(config_.path);
  if (!opened.ok()) {
    throw std::runtime_error("mcap replay: cannot open '" + config_.path + "': " + opened.message);
  }
  // A file written by a run that was killed has no summary; the fallback scan
  // reads the data section instead, which is slower and always works.
  impl_->reader.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);

  // Every image topic in the file, so a bad --role can name the alternatives
  // rather than just failing.
  std::vector<std::string> image_topics;
  const mcap::ChannelId kNoChannel = 0;
  mcap::ChannelId chosen_channel = kNoChannel;
  const std::string wanted =
      !config_.topic.empty()
          ? config_.topic
          : (config_.role.empty() ? std::string() : "/" + config_.role + "/image_raw");

  for (const auto& [id, channel] : impl_->reader.channels()) {
    const auto schema = impl_->reader.schema(channel->schemaId);
    if (!schema || schema->name != kImageSchema) continue;
    image_topics.push_back(channel->topic);
    if (chosen_channel == kNoChannel && (wanted.empty() || channel->topic == wanted)) {
      chosen_channel = id;
      topic_ = channel->topic;
    }
  }

  if (image_topics.empty()) {
    throw std::runtime_error("mcap replay: '" + config_.path + "' holds no " +
                             std::string(kImageSchema) + " topic to replay");
  }
  if (chosen_channel == kNoChannel) {
    std::string available;
    std::sort(image_topics.begin(), image_topics.end());
    for (const std::string& t : image_topics) available += "\n    " + t;
    throw std::runtime_error("mcap replay: no topic '" + wanted + "' in '" + config_.path +
                             "'. Image topics here:" + available);
  }

  if (const auto& stats = impl_->reader.statistics()) {
    const auto it = stats->channelMessageCounts.find(chosen_channel);
    if (it != stats->channelMessageCounts.end()) message_count_ = it->second;
  }

  // The geometry is the first message's own, because a sensor_msgs/Image
  // describes itself -- nothing outside the file has to be told what is in it.
  DecodedImage first;
  bool have_first = false;
  mcap::ReadMessageOptions options;
  options.readOrder = mcap::ReadMessageOptions::ReadOrder::LogTimeOrder;
  options.topicFilter = [this](std::string_view t) { return t == topic_; };
  for (const auto& message : impl_->reader.readMessages([](const mcap::Status&) {}, options)) {
    have_first = decode_image(message.message.data, message.message.dataSize, first);
    break;
  }
  if (!have_first) {
    throw std::runtime_error("mcap replay: the first message on '" + topic_ +
                             "' is not a decodable sensor_msgs/Image");
  }

  const char* format = camera_format_of(first.encoding);
  if (format == nullptr) {
    throw std::runtime_error("mcap replay: '" + topic_ + "' is encoded '" + first.encoding +
                             "', which this pipeline has no pixel format for");
  }

  frame_id_ = first.frame_id;
  geometry_.width = first.width;
  geometry_.height = first.height;
  geometry_.stride_bytes = first.step;
  geometry_.pixel_format = format;
  geometry_.frame_bytes = first.bytes;
  // No transport padding to allow for: what the file holds is exactly what a
  // slot has to take.
  geometry_.buffer_bytes = first.bytes;
}

McapReplaySource::~McapReplaySource() { stop(); }

void McapReplaySource::start(FrameSink& sink) {
  if (running_.load(std::memory_order_relaxed)) return;
  if (sink.slot_bytes() < geometry_.buffer_bytes) {
    throw std::runtime_error("mcap replay: sink slots are " + std::to_string(sink.slot_bytes()) +
                             " bytes but a frame is " + std::to_string(geometry_.buffer_bytes));
  }

  held_.assign(sink.slot_count(), false);
  finished_.store(false, std::memory_order_release);
  running_.store(true, std::memory_order_relaxed);
  thread_ = std::thread([this, &sink] { run(sink); });
}

void McapReplaySource::stop() {
  running_.store(false, std::memory_order_relaxed);
  if (thread_.joinable()) thread_.join();
}

void McapReplaySource::finish(std::string failure_reason) {
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

void McapReplaySource::reclaim(FrameSink& sink) {
  for (uint32_t slot = 0; slot < held_.size(); ++slot) {
    if (held_[slot] && sink.consumed(slot)) held_[slot] = false;
  }
}

uint32_t McapReplaySource::acquire_slot(FrameSink& sink, bool& waited) {
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

void McapReplaySource::run(FrameSink& sink) {
  mcap::ReadMessageOptions options;
  options.readOrder = mcap::ReadMessageOptions::ReadOrder::LogTimeOrder;
  options.topicFilter = [this](std::string_view t) { return t == topic_; };

  try {
    do {
      // One origin per pass, in both clocks: `origin` paces the wall-clock
      // wait, `base` is what rebased timestamps are measured from. Captured
      // together so the reported latency comes out at roughly zero rather than
      // at the gap between the two reads.
      const auto origin = std::chrono::steady_clock::now();
      const uint64_t base = host_now_ns();
      uint64_t epoch_ns = 0;
      bool have_epoch = false;
      uint32_t frame_id = 0;

      for (const auto& message :
           impl_->reader.readMessages([](const mcap::Status&) {}, options)) {
        if (!running_.load(std::memory_order_relaxed)) return;

        DecodedImage image;
        if (!decode_image(message.message.data, message.message.dataSize, image) ||
            image.bytes > geometry_.buffer_bytes) {
          // A message that will not decode, or one bigger than the geometry the
          // first frame established. Counted rather than fatal: the rest of the
          // file is still worth replaying.
          undecodable_.fetch_add(1, std::memory_order_relaxed);
          continue;
        }

        if (!have_epoch) {
          epoch_ns = image.stamp_ns;
          have_epoch = true;
        }

        // Pacing is from the frame's own offset into the recording, never from
        // "previous frame plus a nominal period", so a dropout replays as a
        // stall of exactly the right length.
        const uint64_t offset_ns =
            image.stamp_ns >= epoch_ns
                ? static_cast<uint64_t>(static_cast<double>(image.stamp_ns - epoch_ns) /
                                        config_.speed)
                : 0;
        const auto due = origin + std::chrono::nanoseconds(offset_ns);

        for (;;) {
          if (!running_.load(std::memory_order_relaxed)) return;
          const auto now = std::chrono::steady_clock::now();
          if (now >= due) break;
          std::this_thread::sleep_for(
              std::min(std::chrono::duration_cast<std::chrono::nanoseconds>(due - now),
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

        // Out of the message and into the sink's slot, once. The message is the
        // reader's buffer and is gone at the next iteration, so this cannot be
        // deferred the way a device DMA could.
        std::memcpy(sink.buffers()[slot], image.pixels, image.bytes);

        FrameMeta meta;
        meta.timestamp_ns = config_.rebase_timestamps ? base + offset_ns : image.stamp_ns;
        meta.host_recv_ns = host_now_ns();
        meta.frame_id = frame_id++;
        meta.bytes = image.bytes;

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

std::string McapReplaySource::counters() const {
  std::ostringstream out;
  out << "late=" << late() << " slot_drops=" << slot_drops() << " undecodable=" << undecodable()
      << " loops=" << loops();
  return out.str();
}

std::string McapReplaySource::notes() const {
  std::string note;
  if (slot_drops() > 0) {
    note += " -- slot_drops means the pipeline is slower than the recorded rate";
  }
  if (undecodable() > 0) {
    note += " -- undecodable messages on " + topic_ + " were skipped";
  }
  return note;
}

std::string McapReplaySource::ptp_status() {
  // Labelled "recorded:" so acquire_main prints it as provenance rather than as
  // a live lock state: there is no clock being disciplined here.
  return "recorded:mcap " + topic_;
}

}  // namespace perception
