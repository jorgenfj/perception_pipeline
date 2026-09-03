#include "mcap_player.hpp"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <mcap/mcap.hpp>

#include "frame_sink.hpp"

namespace perception {
namespace {

// Never sleep longer than this in one go, so stop() is acted on promptly even
// when the next message is a long dropout away.
constexpr auto kMaxSleep = std::chrono::milliseconds(20);

constexpr std::string_view kImageSchema = "sensor_msgs/msg/Image";

// The recorder writes one of these; reading it back is how a replay can say
// what clock the stamps it is emitting came from.
constexpr const char* kProvenanceRecord = "perception";

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("mcap replay: " + what);
}

std::string listed(const std::vector<std::string>& names) {
  if (names.empty()) return " (nothing)";
  std::string out;
  for (const std::string& name : names) out += "\n    " + name;
  return out;
}

// Adapts a plain function to the subscriber interface. Accepts everything: a
// callback that wanted to refuse a message would have been written as a class.
class CallbackSubscriber final : public ReplaySubscriber {
 public:
  explicit CallbackSubscriber(std::function<void(const ReplayMessage&)> callback)
      : callback_(std::move(callback)) {}

  bool on_message(const ReplayMessage& message) override {
    callback_(message);
    return true;
  }

 private:
  std::function<void(const ReplayMessage&)> callback_;
};

}  // namespace

// The reader has to outlive every message view taken from it, and mcap.hpp is
// not something the header should drag into every translation unit that only
// wants to construct one of these.
struct McapPlayer::Impl {
  mcap::McapReader reader;

  // The window `start_seconds` / `duration_seconds` resolved into the file's
  // own log times, so the read options can be rebuilt identically on each loop.
  mcap::Timestamp start_time = 0;
  mcap::Timestamp end_time = mcap::MaxTime;

  // Time order needs message indexes, which live in the summary. See the note
  // where this is decided.
  mcap::ReadMessageOptions::ReadOrder read_order =
      mcap::ReadMessageOptions::ReadOrder::LogTimeOrder;

  mcap::ReadMessageOptions options(std::function<bool(std::string_view)> filter) const {
    mcap::ReadMessageOptions out;
    out.readOrder = read_order;
    out.startTime = start_time;
    out.endTime = end_time;
    out.topicFilter = std::move(filter);
    return out;
  }
};

McapPlayer::McapPlayer(const Config& config) : config_(config), impl_(std::make_unique<Impl>()) {
  if (config_.speed <= 0.0) fail("speed must be positive");
  if (config_.start_seconds < 0.0) fail("start_seconds cannot be negative");
  if (config_.duration_seconds < 0.0) fail("duration_seconds cannot be negative");

  const mcap::Status opened = impl_->reader.open(config_.path);
  if (!opened.ok()) fail("cannot open '" + config_.path + "': " + opened.message);

  // A file written by a run that was killed has no summary; the fallback scan
  // reads the data section instead, which is slower and always works.
  (void)impl_->reader.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);

  // --- what is in the file --------------------------------------------------
  std::vector<std::string> present;
  std::vector<std::string> image_present;
  std::map<std::string, ChannelInfo, std::less<>> found;

  for (const auto& [id, channel] : impl_->reader.channels()) {
    const auto schema = impl_->reader.schema(channel->schemaId);

    ChannelInfo info;
    info.topic = channel->topic;
    info.schema_name = schema ? schema->name : std::string();
    info.message_encoding = channel->messageEncoding;
    if (const auto& stats = impl_->reader.statistics()) {
      const auto it = stats->channelMessageCounts.find(id);
      if (it != stats->channelMessageCounts.end()) info.message_count = it->second;
    }

    present.push_back(channel->topic);
    if (info.schema_name == kImageSchema) image_present.push_back(channel->topic);
    found.emplace(channel->topic, std::move(info));
  }

  if (present.empty()) fail("'" + config_.path + "' holds no channel to replay");
  std::sort(present.begin(), present.end());
  std::sort(image_present.begin(), image_present.end());

  // --- which of it feeds the pipeline ---------------------------------------
  // Validated here, but only *selected* below: an explicit list is an override
  // that forces its topics in, while an omitted one follows the filters, so
  // narrowing `topics` to one eye replays as a mono run rather than quietly
  // dragging the other eye back in.
  for (const std::string& topic : config_.image_topics) {
    const auto it = found.find(topic);
    if (it == found.end()) {
      fail("source.image_topics names '" + topic + "', which is not in '" + config_.path +
           "'. Image topics here:" + listed(image_present));
    }
    if (it->second.schema_name != kImageSchema) {
      fail("source.image_topics names '" + topic + "', which carries '" + it->second.schema_name +
           "' rather than " + std::string(kImageSchema));
    }
    if (std::find(config_.exclude.begin(), config_.exclude.end(), topic) !=
        config_.exclude.end()) {
      fail("'" + topic +
           "' is in both source.image_topics and source.exclude; a topic cannot be both fed to "
           "the pipeline and dropped");
    }
  }

  // --- which of it is replayed ----------------------------------------------
  std::vector<std::string> selected;
  if (config_.topics.empty()) {
    selected = present;
  } else {
    for (const std::string& topic : config_.topics) {
      if (found.find(topic) == found.end()) {
        fail("source.topics names '" + topic + "', which is not in '" + config_.path +
             "'. Topics here:" + listed(present));
      }
      if (std::count(config_.topics.begin(), config_.topics.end(), topic) > 1) {
        fail("source.topics lists '" + topic + "' twice");
      }
      selected.push_back(topic);
    }
  }

  for (const std::string& topic : config_.exclude) {
    selected.erase(std::remove(selected.begin(), selected.end(), topic), selected.end());
  }

  // An explicit list is not optional: it is what this replay is for, so it goes
  // in whatever the filters said.
  for (const std::string& topic : config_.image_topics) {
    if (std::find(selected.begin(), selected.end(), topic) == selected.end()) {
      selected.push_back(topic);
    }
  }

  // Sorted rather than file order when the config does not say, so "/left/..."
  // is stream 0 and a run is reproducible across two recordings of one rig.
  image_topics_ = config_.image_topics;
  if (image_topics_.empty()) {
    for (const std::string& topic : selected) {
      if (found.find(topic)->second.schema_name == kImageSchema) image_topics_.push_back(topic);
    }
    std::sort(image_topics_.begin(), image_topics_.end());
  }

  if (selected.empty()) {
    fail("the filters leave nothing to replay from '" + config_.path +
         "'. Topics here:" + listed(present));
  }

  std::sort(selected.begin(), selected.end());
  for (const std::string& topic : selected) {
    const ChannelInfo& info = found.find(topic)->second;
    channels_.push_back(info);
    auto entry = std::make_unique<Route>();
    entry->info = info;
    routes_.emplace(topic, std::move(entry));
  }

  // --- the window -----------------------------------------------------------
  if (const auto& stats = impl_->reader.statistics()) {
    impl_->start_time = stats->messageStartTime;
    impl_->end_time = stats->messageEndTime == 0 ? mcap::MaxTime : stats->messageEndTime + 1;
  }
  const auto to_ns = [](double seconds) {
    return static_cast<mcap::Timestamp>(seconds * 1e9);
  };
  if (config_.start_seconds > 0.0) impl_->start_time += to_ns(config_.start_seconds);
  if (config_.duration_seconds > 0.0) {
    impl_->end_time = impl_->start_time + to_ns(config_.duration_seconds);
  }

  // --- how it can be read ---------------------------------------------------
  // Time order needs message indexes, and those live in the summary. A file
  // from a run that was killed has no summary at all: the fallback scan above
  // recovers the channels and the statistics but not the indexes, and asking
  // for time order then yields NOTHING -- silently, because the reader reports
  // it through a callback rather than a throw.
  //
  // Such a file can still be read in the order it was written, which is the
  // order the recorder's writer thread drained its queues -- arrival order.
  // Pacing is from each message's own log time either way, so the replay is the
  // same; what is given up is the guarantee for a file some other writer
  // interleaved differently.
  //
  // The fallback scan does rebuild a chunk index -- it sees the Chunk records
  // going past -- but with no message index offsets in it, so the presence of
  // chunk indexes is not the question. Whether they point at anything is.
  const auto& chunks = impl_->reader.chunkIndexes();
  const bool indexed =
      !chunks.empty() && std::all_of(chunks.begin(), chunks.end(), [](const mcap::ChunkIndex& c) {
        return !c.messageIndexOffsets.empty();
      });
  if (!indexed) {
    impl_->read_order = mcap::ReadMessageOptions::ReadOrder::FileOrder;
    unindexed_ = true;
  }

  // --- provenance -----------------------------------------------------------
  // The recorder's note about what clock these stamps are on. Best effort: a
  // file from something else simply has no such record, and replay does not
  // depend on it.
  for (const auto& [name, index] : impl_->reader.metadataIndexes()) {
    if (name != kProvenanceRecord) continue;
    mcap::TypedRecordReader records(*impl_->reader.dataSource(), index.offset,
                                    index.offset + index.length);
    records.onMetadata = [this](const mcap::Metadata& metadata, mcap::ByteOffset) {
      const auto epoch = metadata.metadata.find("timestamp_epoch");
      const auto offset = metadata.metadata.find("epoch_offset_ns");
      if (epoch == metadata.metadata.end()) return;
      recorded_clock_ = epoch->second;
      if (offset != metadata.metadata.end()) recorded_clock_ += " offset=" + offset->second + "ns";
    };
    while (records.next()) {
    }
    break;
  }
}

McapPlayer::~McapPlayer() { stop(); }

void McapPlayer::note_problem(const std::string& what) {
  if (problem_.empty()) problem_ = what;
}

McapPlayer::Route& McapPlayer::route(std::string_view topic) {
  const auto it = routes_.find(topic);
  if (it == routes_.end()) {
    std::vector<std::string> replayed;
    for (const auto& [name, entry] : routes_) replayed.push_back(name);
    fail("'" + std::string(topic) + "' is not being replayed. Replaying:" + listed(replayed));
  }
  return *it->second;
}

const McapPlayer::Route& McapPlayer::route(std::string_view topic) const {
  return const_cast<McapPlayer*>(this)->route(topic);
}

bool McapPlayer::read_first(std::string_view topic,
                            const std::function<void(const ReplayMessage&)>& body) {
  const Route& selected = route(topic);

  const std::string wanted(topic);
  const mcap::ReadMessageOptions options =
      impl_->options([&wanted](std::string_view candidate) { return candidate == wanted; });

  for (const auto& message : impl_->reader.readMessages(
           [this](const mcap::Status& status) { note_problem(status.message); }, options)) {
    ReplayMessage out;
    out.topic = selected.info.topic;
    out.schema_name = selected.info.schema_name;
    out.message_encoding = selected.info.message_encoding;
    out.log_time_ns = message.message.logTime;
    out.stamp_ns = message.message.publishTime != 0 ? message.message.publishTime
                                                    : message.message.logTime;
    out.sequence = message.message.sequence;
    out.data = message.message.data;
    out.size = message.message.dataSize;
    body(out);
    return true;
  }
  return false;
}

void McapPlayer::subscribe(std::string_view topic, ReplaySubscriber& subscriber) {
  Route& selected = route(topic);
  if (selected.subscriber != nullptr) {
    fail("'" + std::string(topic) + "' already has a subscriber; a topic is routed once");
  }
  selected.subscriber = &subscriber;
}

void McapPlayer::subscribe(std::string_view topic,
                           std::function<void(const ReplayMessage&)> callback) {
  Route& selected = route(topic);
  if (selected.subscriber != nullptr) {
    fail("'" + std::string(topic) + "' already has a subscriber; a topic is routed once");
  }
  selected.owned = std::make_unique<CallbackSubscriber>(std::move(callback));
  selected.subscriber = selected.owned.get();
}

void McapPlayer::expect_bind() { pending_binds_.fetch_add(1, std::memory_order_relaxed); }

void McapPlayer::bind_done() {
  if (pending_binds_.fetch_sub(1, std::memory_order_acq_rel) == 1) start();
}

void McapPlayer::add_finished_callback(std::function<void()> callback) {
  on_finished_.push_back(std::move(callback));
}

void McapPlayer::start() {
  if (started_) return;
  started_ = true;
  finished_.store(false, std::memory_order_release);
  running_.store(true, std::memory_order_relaxed);
  thread_ = std::thread([this] { run(); });
}

void McapPlayer::stop() {
  running_.store(false, std::memory_order_relaxed);
  if (thread_.joinable()) thread_.join();
}

void McapPlayer::finish(std::string failure_reason) {
  if (!failure_reason.empty()) {
    failure_ = std::move(failure_reason);
    failed_.store(true, std::memory_order_release);
  }
  finished_.store(true, std::memory_order_release);
  for (const auto& callback : on_finished_) {
    try {
      callback();
    } catch (...) {
    }
  }
}

void McapPlayer::run() {
  // One predicate over every selected channel, so the whole file comes off a
  // single merged iteration and the recorded interleaving survives.
  const mcap::ReadMessageOptions options = impl_->options(
      [this](std::string_view topic) { return routes_.find(topic) != routes_.end(); });

  try {
    do {
      // One origin per pass, in both clocks. `origin` paces the wall-clock wait
      // against recorded log time; `base` is what emitted stamps are measured
      // from. Captured together so the reported latency comes out at roughly
      // zero rather than at the gap between the two reads.
      const auto origin = std::chrono::steady_clock::now();
      const uint64_t base = host_now_ns();
      uint64_t epoch_log = 0;
      uint64_t epoch_stamp = 0;
      bool have_epoch = false;

      for (const auto& message : impl_->reader.readMessages(
               [this](const mcap::Status& status) { note_problem(status.message); }, options)) {
        if (!running_.load(std::memory_order_relaxed)) return;

        const auto it = routes_.find(message.channel->topic);
        if (it == routes_.end()) continue;  // topicFilter should have caught it
        Route& selected = *it->second;

        const uint64_t log_time = message.message.logTime;
        const uint64_t stamp = message.message.publishTime != 0 ? message.message.publishTime
                                                                : message.message.logTime;
        if (!have_epoch) {
          epoch_log = log_time;
          epoch_stamp = stamp;
          have_epoch = true;
        }

        // Pacing is from this message's own offset into the recording, never
        // from "previous message plus a nominal period", so a dropout replays
        // as a stall of exactly the right length and a slow subscriber is
        // caught up rather than drifting.
        const uint64_t offset_ns =
            log_time >= epoch_log
                ? static_cast<uint64_t>(static_cast<double>(log_time - epoch_log) / config_.speed)
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

        if (selected.subscriber == nullptr) {
          selected.unrouted.fetch_add(1, std::memory_order_relaxed);
          continue;
        }

        ReplayMessage out;
        out.topic = selected.info.topic;
        out.schema_name = selected.info.schema_name;
        out.message_encoding = selected.info.message_encoding;
        out.log_time_ns = log_time;
        // One affine map on the message's OWN stamp, not on log time: that is
        // what preserves every cross-topic relationship -- the stereo skew
        // above all -- through a file whose log and publish clocks differ.
        // Dividing by speed keeps a fast replay from reading as latency.
        // Clamped at the epoch rather than wrapping: a message stamped before
        // the first one is possible whenever log and publish order disagree, and
        // an unsigned difference there would come back as half a millennium of
        // wait rather than a small negative.
        const uint64_t since_epoch =
            stamp >= epoch_stamp
                ? static_cast<uint64_t>(static_cast<double>(stamp - epoch_stamp) / config_.speed)
                : 0;
        out.stamp_ns = config_.rebase_timestamps ? base + since_epoch : stamp;
        out.sequence = message.message.sequence;
        out.data = message.message.data;
        out.size = message.message.dataSize;

        // The subscriber is allowed to take its time -- an image sink waits up
        // to slot_wait_ms for a free slot -- and that wait is the only thing
        // that blocks the timeline. It is bounded by the subscriber, which is
        // the only party that knows what it is waiting for.
        if (selected.subscriber->on_message(out)) {
          selected.delivered.fetch_add(1, std::memory_order_relaxed);
        } else {
          selected.dropped.fetch_add(1, std::memory_order_relaxed);
        }
      }

      if (config_.loop && running_.load(std::memory_order_relaxed)) {
        loops_.fetch_add(1, std::memory_order_relaxed);
      }
    } while (config_.loop && running_.load(std::memory_order_relaxed));
  } catch (const std::exception& error) {
    finish(error.what());
    return;
  }

  // Played to the end with looping off. Not a failure: the run is simply over,
  // and the owner is told so it stops waiting for a publish that is not coming.
  finish({});
}

uint64_t McapPlayer::delivered(std::string_view topic) const {
  return route(topic).delivered.load(std::memory_order_relaxed);
}

uint64_t McapPlayer::dropped(std::string_view topic) const {
  return route(topic).dropped.load(std::memory_order_relaxed);
}

uint64_t McapPlayer::unrouted(std::string_view topic) const {
  return route(topic).unrouted.load(std::memory_order_relaxed);
}

std::string McapPlayer::counters() const {
  uint64_t dropped_total = 0;
  uint64_t unrouted_total = 0;
  for (const auto& [name, entry] : routes_) {
    dropped_total += entry->dropped.load(std::memory_order_relaxed);
    unrouted_total += entry->unrouted.load(std::memory_order_relaxed);
  }
  std::ostringstream out;
  out << "loops=" << loops() << " dropped=" << dropped_total << " unrouted=" << unrouted_total;
  return out.str();
}

std::string McapPlayer::health_line() const {
  std::ostringstream out;
  out << "replay " << config_.path;
  if (unindexed_) out << " (unindexed, file order)";
  if (!problem_.empty()) out << " [reader: " << problem_ << "]";
  for (const auto& [name, entry] : routes_) {
    out << " | " << name << " " << entry->delivered.load(std::memory_order_relaxed);
    if (entry->info.message_count > 0) out << "/" << entry->info.message_count;
    const uint64_t dropped_here = entry->dropped.load(std::memory_order_relaxed);
    if (dropped_here > 0) out << " dropped=" << dropped_here;
    if (entry->subscriber == nullptr) out << " (unrouted)";
  }
  return out.str();
}

}  // namespace perception
