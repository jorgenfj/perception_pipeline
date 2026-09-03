#include "image_replay_source.hpp"

#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "cdr_reader.hpp"

namespace perception {
namespace {

const char* camera_format_of(const std::string& ros_encoding) {
  static const std::unordered_map<std::string, const char*> kFormats = {
      {"bayer_rggb8", "BayerRG8"}, {"bayer_grbg8", "BayerGR8"}, {"bayer_gbrg8", "BayerGB8"},
      {"bayer_bggr8", "BayerBG8"}, {"mono8", "Mono8"},          {"rgb8", "RGB8"},
      {"rgba8", "RGBa8"},
  };
  const auto it = kFormats.find(ros_encoding);
  return it == kFormats.end() ? nullptr : it->second;
}

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

ImageReplaySource::ImageReplaySource(McapPlayer& player, std::string topic)
    : player_(player), topic_(std::move(topic)), slot_wait_ms_(player.slot_wait_ms()) {
  for (const McapPlayer::ChannelInfo& channel : player_.channels()) {
    if (channel.topic == topic_) message_count_ = channel.message_count;
  }

  DecodedImage first;
  bool decoded = false;
  const bool present = player_.read_first(topic_, [&](const ReplayMessage& message) {
    decoded = decode_image(message.data, message.size, first);
  });

  if (!present) {
    throw std::runtime_error(
        "mcap replay: '" + topic_ + "' carries no message to replay" +
        (player_.problem().empty() ? "" : " -- the reader said: " + player_.problem()));
  }
  if (!decoded) {
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
  geometry_.buffer_bytes = first.bytes;

  player_.subscribe(topic_, *this);
  player_.expect_bind();
}

ImageReplaySource::~ImageReplaySource() = default;

void ImageReplaySource::set_finished_callback(std::function<void()> cb) {
  player_.add_finished_callback(std::move(cb));
}

void ImageReplaySource::start(FrameSink& sink) {
  if (sink_ != nullptr) return;
  if (sink.slot_bytes() < geometry_.buffer_bytes) {
    throw std::runtime_error("mcap replay: '" + topic_ + "' sink slots are " +
                             std::to_string(sink.slot_bytes()) + " bytes but a frame is " +
                             std::to_string(geometry_.buffer_bytes));
  }

  held_.assign(sink.slot_count(), false);
  sink_ = &sink;
  player_.bind_done();
}

void ImageReplaySource::stop() { player_.stop(); }

void ImageReplaySource::reclaim() {
  for (uint32_t slot = 0; slot < held_.size(); ++slot) {
    if (held_[slot] && sink_->consumed(slot)) held_[slot] = false;
  }
}

uint32_t ImageReplaySource::acquire_slot(bool& waited) {
  waited = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(slot_wait_ms_);
  for (;;) {
    reclaim();
    for (uint32_t slot = 0; slot < held_.size(); ++slot) {
      if (!held_[slot]) return slot;
    }
    if (!player_.running()) return FrameSink::kNoSlot;
    if (std::chrono::steady_clock::now() >= deadline) return FrameSink::kNoSlot;
    waited = true;
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
}

bool ImageReplaySource::on_message(const ReplayMessage& message) {

  if (sink_ == nullptr) return false;

  DecodedImage image;
  if (!decode_image(message.data, message.size, image) || image.bytes > geometry_.buffer_bytes) {
    undecodable_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  bool waited = false;
  const uint32_t slot = acquire_slot(waited);
  if (waited) late_.fetch_add(1, std::memory_order_relaxed);
  if (slot == FrameSink::kNoSlot) return false;

  std::memcpy(sink_->buffers()[slot], image.pixels, image.bytes);

  FrameMeta meta;
  meta.timestamp_ns = message.stamp_ns;
  meta.host_recv_ns = host_now_ns();
  meta.frame_id = next_frame_id_++;
  meta.bytes = image.bytes;

  held_[slot] = true;
  sink_->commit(slot, meta);
  return true;
}

std::string ImageReplaySource::counters() const {
  std::ostringstream out;
  out << "late=" << late() << " slot_drops=" << slot_drops() << " undecodable=" << undecodable()
      << " loops=" << player_.loops();
  return out.str();
}

std::string ImageReplaySource::notes() const {
  std::string note;
  if (slot_drops() > 0) {
    note += " -- slot_drops means the pipeline is slower than the recorded rate";
  }
  if (undecodable() > 0) {
    note += " -- undecodable messages on " + topic_ + " were skipped";
  }
  if (player_.unindexed()) {
    note +=
        " -- this recording has no summary (the run that wrote it was killed), so it is replayed "
        "in the order it was written rather than in log-time order";
  }
  return note;
}

std::string ImageReplaySource::ptp_status() {
  const std::string& clock = player_.recorded_clock();
  return "recorded:mcap " + topic_ + (clock.empty() ? "" : " " + clock);
}

}  // namespace perception
