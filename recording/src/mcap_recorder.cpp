#include "mcap_recorder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <stdexcept>
#include <utility>

#define MCAP_IMPLEMENTATION
#include <mcap/mcap.hpp>

namespace perception {
namespace {

// Long enough that an idle recorder costs nothing, short enough that the
// periodic chunk flush lands within a small fraction of flush_seconds.
constexpr std::chrono::milliseconds kIdlePoll{100};

std::string timestamped_name() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[64];
  std::strftime(buf, sizeof(buf), "recording-%Y-%m-%dT%H-%M-%S.mcap", &tm);
  return buf;
}

uint64_t steady_us() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}

}  // namespace

struct McapRecorder::Impl {
  mcap::McapWriter writer;

  struct Schema {
    std::string type;
    std::string text;
    mcap::SchemaId id = 0;
  };
  std::vector<Schema> schemas;
};

McapRecorder::McapRecorder(const Config& config)
    : config_(config), impl_(std::make_unique<Impl>()) {
  std::filesystem::create_directories(config_.root);
  path_ = (std::filesystem::path(config_.root) / timestamped_name()).string();


  mcap::McapWriterOptions options("ros2");
  options.compression = config_.compress ? mcap::Compression::Zstd : mcap::Compression::None;
  options.chunkSize = static_cast<uint64_t>(config_.chunk_mb) * 1024 * 1024;

  const mcap::Status status = impl_->writer.open(path_, options);
  if (!status.ok()) {
    throw std::runtime_error("mcap: cannot open '" + path_ + "': " + status.message);
  }

  mcap::Metadata meta;
  meta.name = "perception";
  meta.metadata["timestamp_epoch"] = "TAI";
  meta.metadata["timestamp_source"] = "camera PTP clock (GevTimestamp), verbatim";
  meta.metadata["note"] =
      "log_time, publish_time and header.stamp are all the camera's own PTP clock, which counts "
      "TAI. CLOCK_REALTIME counts UTC and is currently 37s behind.";
  const mcap::Status meta_status = impl_->writer.write(meta);
  if (!meta_status.ok()) {
    std::printf("mcap: warning -- could not write the epoch metadata (%s); timestamps are TAI\n",
                meta_status.message.c_str());
  }
}

McapRecorder::~McapRecorder() { close(); }

McapRecorder::TopicId McapRecorder::add_topic(const Topic& topic) {
  if (started_) {
    throw std::runtime_error("mcap: add_topic('" + topic.name + "') after start()");
  }
  if (topic.max_message_bytes == 0) {
    throw std::runtime_error("mcap: topic '" + topic.name + "' declares no max_message_bytes");
  }

  mcap::SchemaId schema_id = 0;
  const auto known = std::find_if(impl_->schemas.begin(), impl_->schemas.end(),
                                  [&](const Impl::Schema& s) { return s.type == topic.type; });
  if (known != impl_->schemas.end()) {
    if (known->text != topic.schema) {
      throw std::runtime_error("mcap: '" + topic.type +
                               "' was already declared with a different definition -- a file "
                               "carrying two definitions of one type is one nobody can trust");
    }
    schema_id = known->id;
  } else {
    mcap::Schema schema(topic.type, "ros2msg", topic.schema);
    impl_->writer.addSchema(schema);
    impl_->schemas.push_back(Impl::Schema{topic.type, topic.schema, schema.id});
    schema_id = schema.id;
  }

  mcap::Channel channel(topic.name, "cdr", schema_id);
  impl_->writer.addChannel(channel);

  auto state = std::make_unique<TopicState>();
  state->declared = topic;
  state->channel_id = static_cast<uint32_t>(channel.id);

  // buffer_seconds of it, floored so a slow topic still has room and capped so
  // a large message cannot take the whole budget.
  if (topic.queue_depth > 0) {
    state->depth = topic.queue_depth;
  } else {
    const double wanted = std::ceil(topic.rate_hz * config_.buffer_seconds);
    uint32_t depth = wanted > 0.0 ? static_cast<uint32_t>(wanted) : 0u;
    depth = std::max(depth, 4u);
    const std::size_t budget = static_cast<std::size_t>(config_.topic_memory_mb) * 1024 * 1024;
    const uint32_t cap = static_cast<uint32_t>(budget / topic.max_message_bytes);
    state->depth = std::max(1u, std::min(depth, cap));
  }

  topics_.push_back(std::move(state));
  return static_cast<TopicId>(topics_.size() - 1);
}

void McapRecorder::start() {
  if (started_) return;
  if (topics_.empty()) {
    throw std::runtime_error("mcap: start() with no topics -- there would be nothing to write");
  }

  std::size_t reserved = 0;
  for (const std::unique_ptr<TopicState>& state : topics_) {
    // depth + 1: see TopicState::spare. Reserved now so the first message of a
    // run does not allocate on the thread that produced it.
    state->spare.reserve(state->depth + 1);
    for (uint32_t i = 0; i <= state->depth; ++i) {
      CdrWriter cdr;
      cdr.reserve(state->declared.max_message_bytes);
      state->spare.push_back(std::move(cdr));
    }
    reserved += static_cast<std::size_t>(state->depth + 1) * state->declared.max_message_bytes;
  }

  started_ = true;
  running_ = true;
  thread_ = std::thread([this] { run(); });

  std::printf("mcap: recording -> %s (%zu topics, %.1fMB of queue buffers)\n", path_.c_str(),
              topics_.size(), static_cast<double>(reserved) / 1e6);
}

bool McapRecorder::take_spare(TopicId topic, CdrWriter& out) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (topic >= topics_.size()) return false;

  TopicState& state = *topics_[topic];
  if (!running_ || state.queued >= state.depth) {
    state.drops.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (state.spare.empty()) {
    // Unreachable with one pushing thread per topic, which is the contract:
    // depth + 1 buffers exist and the check above caps what can be out at depth.
    state.drops.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  out = std::move(state.spare.back());
  state.spare.pop_back();
  return true;
}

bool McapRecorder::enqueue(TopicId topic, uint64_t timestamp_ns, CdrWriter&& cdr) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    TopicState& state = *topics_[topic];

    // close() can land between take_spare() and here. The buffer still goes
    // back, or one late message would cost the topic every one after it.
    if (!running_) {
      state.drops.fetch_add(1, std::memory_order_relaxed);
      cdr.clear();
      state.spare.push_back(std::move(cdr));
      return false;
    }

    bytes_in_flight_ += cdr.size();
    if (bytes_in_flight_ > bytes_in_flight_peak_.load(std::memory_order_relaxed)) {
      bytes_in_flight_peak_.store(bytes_in_flight_, std::memory_order_relaxed);
    }

    queue_.push_back(Message{topic, timestamp_ns, std::move(cdr)});
    ++state.queued;
    if (state.queued > state.peak.load(std::memory_order_relaxed)) {
      state.peak.store(state.queued, std::memory_order_relaxed);
    }
  }
  queued_.notify_one();
  return true;
}

void McapRecorder::recycle(TopicId topic, CdrWriter&& cdr) {
  cdr.clear();  // capacity survives, which is the whole point of the pool
  const std::lock_guard<std::mutex> lock(mutex_);
  topics_[topic]->spare.push_back(std::move(cdr));
}

void McapRecorder::run() {
  const auto interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(std::max(0.0, config_.flush_seconds)));
  auto next_flush = std::chrono::steady_clock::now() + interval;

  for (;;) {
    Message message;
    bool have = false;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      queued_.wait_for(lock, kIdlePoll, [this] { return !queue_.empty() || !running_; });

      if (!queue_.empty()) {
        message = std::move(queue_.front());
        queue_.pop_front();
        bytes_in_flight_ -= message.cdr.size();
        --topics_[message.topic]->queued;
        have = true;
      } else if (!running_) {
        // Drain before exiting: a close() that threw away queued messages would
        // lose the tail of every recording.
        break;
      }
    }

    if (have) {
      TopicState& state = *topics_[message.topic];

      mcap::Message out;
      out.channelId = static_cast<mcap::ChannelId>(state.channel_id);
      out.sequence = state.sequence++;
      out.logTime = message.timestamp_ns;
      out.publishTime = message.timestamp_ns;
      out.data = reinterpret_cast<const std::byte*>(message.cdr.data().data());
      out.dataSize = message.cdr.size();

      const uint64_t started = steady_us();
      const mcap::Status status = impl_->writer.write(out);
      const uint64_t took = steady_us() - started;
      if (took > write_max_us_.load(std::memory_order_relaxed)) {
        write_max_us_.store(took, std::memory_order_relaxed);
      }

      if (status.ok()) {
        state.written.fetch_add(1, std::memory_order_relaxed);
        bytes_written_.fetch_add(message.cdr.size(), std::memory_order_relaxed);
      } else {
        // Latch and keep draining: the alternative is a thread that dies with
        // the queue full and a producer that never learns why.
        state.drops.fetch_add(1, std::memory_order_relaxed);
      }

      recycle(message.topic, std::move(message.cdr));
    }

    if (config_.flush_seconds > 0.0 && std::chrono::steady_clock::now() >= next_flush) {
      // Bounds crash exposure in time rather than in bytes. Does nothing when
      // no chunk is open, so an idle recorder pays nothing for it.
      impl_->writer.closeLastChunk();
      next_flush = std::chrono::steady_clock::now() + interval;
    }
  }

  impl_->writer.closeLastChunk();
}

void McapRecorder::close() {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return;
    closed_ = true;
    running_ = false;
  }
  queued_.notify_all();
  if (thread_.joinable()) thread_.join();

  impl_->writer.close();
  std::printf("mcap: stopped %s (%s)\n", path_.c_str(), health_line().c_str());
}

uint64_t McapRecorder::written() const {
  uint64_t total = 0;
  for (const std::unique_ptr<TopicState>& state : topics_) {
    total += state->written.load(std::memory_order_relaxed);
  }
  return total;
}

uint64_t McapRecorder::drops() const {
  uint64_t total = 0;
  for (const std::unique_ptr<TopicState>& state : topics_) {
    total += state->drops.load(std::memory_order_relaxed);
  }
  return total;
}

uint64_t McapRecorder::written(TopicId topic) const {
  return topic < topics_.size() ? topics_[topic]->written.load(std::memory_order_relaxed) : 0;
}

uint64_t McapRecorder::drops(TopicId topic) const {
  return topic < topics_.size() ? topics_[topic]->drops.load(std::memory_order_relaxed) : 0;
}

uint64_t McapRecorder::bytes_in_flight() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return bytes_in_flight_;
}

std::string McapRecorder::health_line() const {
  std::string drops_list;
  const TopicState* worst = nullptr;
  double worst_share = -1.0;

  for (const std::unique_ptr<TopicState>& state : topics_) {
    if (!drops_list.empty()) drops_list += "/";
    drops_list += std::to_string(state->drops.load(std::memory_order_relaxed));

    const uint32_t peak = state->peak.load(std::memory_order_relaxed);
    const double share = state->depth > 0 ? static_cast<double>(peak) / state->depth : 0.0;
    if (share > worst_share) {
      worst_share = share;
      worst = state.get();
    }
  }
  if (drops_list.empty()) drops_list = "-";

  char line[320];
  std::snprintf(line, sizeof(line),
                "mcap: %llu msgs, drops=%s, peak %u/%u %s, %.1fMB queued (peak %.1fMB), "
                "write_max=%lluus, %.1fMB",
                static_cast<unsigned long long>(written()), drops_list.c_str(),
                worst ? worst->peak.load(std::memory_order_relaxed) : 0u, worst ? worst->depth : 0u,
                worst ? worst->declared.name.c_str() : "-",
                static_cast<double>(bytes_in_flight()) / 1e6,
                static_cast<double>(bytes_in_flight_peak()) / 1e6,
                static_cast<unsigned long long>(write_max_us()),
                static_cast<double>(bytes_written()) / 1e6);
  return line;
}

}  // namespace perception
