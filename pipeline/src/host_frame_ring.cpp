#include "host_frame_ring.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace perception {

// Slot bookkeeping, shared with every frame handed out so a frame may outlive
// the ring that issued it.
struct HostFrameRing::Store {
  std::mutex mutex;
  std::condition_variable frame_ready;

  std::unique_ptr<unsigned char[]> block;

  // The slots' capacity. Not readable off a frame: HostFrame::bytes is what the
  // last publish() into that slot actually wrote, which is the length a
  // consumer needs and may be shorter.
  std::size_t frame_bytes = 0;

  std::vector<HostFrame> frames;
  std::vector<uint32_t> holds;  // consumers currently holding each slot

  // The newest published frame. Never overwritten while it holds that title,
  // so a consumer that has not looked yet still finds something to take.
  uint32_t latest = kNoSlot;
  uint64_t latest_seq = 0;

  uint64_t sequence = 0;
  uint32_t in_use = 0;
  uint32_t peak_in_use = 0;
  bool running = true;

  void release(uint32_t slot) noexcept {
    const std::lock_guard<std::mutex> lock(mutex);
    if (holds[slot] > 0 && --holds[slot] == 0 && slot != latest) --in_use;
  }
};

HostFrameRing::HostFrameRing(uint32_t slots, std::size_t frame_bytes, uint32_t width,
                             uint32_t height)
    : store_(std::make_shared<Store>()) {
  // Two is the floor even with no consumers: one to write into and one holding
  // the newest, or the producer would overwrite what it just published.
  if (slots < 2) throw std::runtime_error("HostFrameRing: needs at least two slots");
  if (frame_bytes == 0) throw std::runtime_error("HostFrameRing: frame_bytes is zero");

  store_->frame_bytes = frame_bytes;
  store_->block = std::make_unique<unsigned char[]>(frame_bytes * slots);
  store_->frames.resize(slots);
  store_->holds.assign(slots, 0);

  for (uint32_t i = 0; i < slots; ++i) {
    HostFrame& frame = store_->frames[i];
    frame.data = store_->block.get() + static_cast<std::size_t>(i) * frame_bytes;
    frame.bytes = frame_bytes;
    frame.width = width;
    frame.height = height;
  }
}

HostFrameRing::~HostFrameRing() { stop(); }

uint32_t HostFrameRing::add_consumer(std::string name) {
  if (publishing_) {
    throw std::runtime_error("HostFrameRing::add_consumer('" + name + "') after publishing began");
  }
  auto consumer = std::make_unique<Consumer>();
  consumer->name = std::move(name);
  consumers_.push_back(std::move(consumer));
  return static_cast<uint32_t>(consumers_.size() - 1);
}

bool HostFrameRing::publish(const void* data, std::size_t bytes, uint64_t timestamp_ns) {
  publishing_ = true;

  uint32_t slot = kNoSlot;
  {
    const std::lock_guard<std::mutex> lock(store_->mutex);
    if (!store_->running) return false;

    // Free means nobody holds it and it is not the newest. Any free slot will
    // do -- insisting on the next one in order would drop a frame whenever a
    // slow consumer happened to be sitting on that index.
    for (uint32_t i = 0; i < store_->frames.size(); ++i) {
      if (store_->holds[i] == 0 && i != store_->latest) {
        slot = i;
        break;
      }
    }
    if (slot == kNoSlot) {
      drops_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    // Reserved for the duration of the copy, which happens outside the lock.
    store_->holds[slot] = 1;
    ++store_->in_use;
    if (store_->in_use > store_->peak_in_use) store_->peak_in_use = store_->in_use;
  }

  HostFrame& frame = store_->frames[slot];
  if (bytes > store_->frame_bytes) {
    const std::lock_guard<std::mutex> lock(store_->mutex);
    store_->holds[slot] = 0;
    --store_->in_use;
    throw std::runtime_error("HostFrameRing::publish: frame larger than the ring was sized for");
  }
  std::memcpy(frame.data, data, bytes);

  {
    const std::lock_guard<std::mutex> lock(store_->mutex);
    frame.timestamp_ns = timestamp_ns;
    frame.bytes = bytes;
    frame.sequence = ++store_->sequence;

    // The slot that was newest stops being reserved; it goes free unless a
    // consumer is holding it.
    const uint32_t previous = store_->latest;
    store_->latest = slot;
    store_->latest_seq = frame.sequence;
    --store_->holds[slot];  // reserved by `latest` from here, not by the copy
    if (previous != kNoSlot && store_->holds[previous] == 0) --store_->in_use;
  }
  store_->frame_ready.notify_all();

  published_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

std::shared_ptr<const HostFrame> HostFrameRing::acquire_latest(uint32_t consumer) {
  if (consumer >= consumers_.size()) return nullptr;
  Consumer& reader = *consumers_[consumer];

  uint32_t slot = kNoSlot;
  uint64_t sequence = 0;
  {
    std::unique_lock<std::mutex> lock(store_->mutex);
    store_->frame_ready.wait(
        lock, [&] { return !store_->running || store_->latest_seq > reader.cursor; });
    if (!store_->running) return nullptr;

    slot = store_->latest;
    sequence = store_->latest_seq;

    // No in_use change: the slot was already unavailable as `latest`, and it
    // stays unavailable now that it is held. The previous hold is released by
    // the caller dropping its old shared_ptr, not here -- releasing it twice is
    // exactly the bug the deleter exists to prevent.
    ++store_->holds[slot];

    // Everything published between the last hand-off and this one.
    reader.skipped.fetch_add(sequence - reader.cursor - 1, std::memory_order_relaxed);
    reader.cursor = sequence;
  }

  return std::shared_ptr<const HostFrame>(&store_->frames[slot],
                                          [store = store_, slot](const HostFrame*) noexcept {
                                            store->release(slot);
                                          });
}

void HostFrameRing::stop() {
  {
    const std::lock_guard<std::mutex> lock(store_->mutex);
    if (!store_->running) return;
    store_->running = false;
  }
  store_->frame_ready.notify_all();
}

uint32_t HostFrameRing::slots() const {
  const std::lock_guard<std::mutex> lock(store_->mutex);
  return static_cast<uint32_t>(store_->frames.size());
}

std::size_t HostFrameRing::frame_bytes() const { return store_->frame_bytes; }

uint64_t HostFrameRing::skipped(uint32_t consumer) const {
  if (consumer >= consumers_.size()) return 0;
  return consumers_[consumer]->skipped.load(std::memory_order_relaxed);
}

uint32_t HostFrameRing::in_use() const {
  const std::lock_guard<std::mutex> lock(store_->mutex);
  return store_->in_use;
}

uint32_t HostFrameRing::peak_in_use() const {
  const std::lock_guard<std::mutex> lock(store_->mutex);
  return store_->peak_in_use;
}

std::string HostFrameRing::health_line() const {
  std::string skips;
  for (const std::unique_ptr<Consumer>& reader : consumers_) {
    skips += " " + reader->name + "=" +
             std::to_string(reader->skipped.load(std::memory_order_relaxed));
  }
  if (skips.empty()) skips = " (none)";

  char line[256];
  std::snprintf(line, sizeof(line), "tap: %llu published, drops=%llu, held %u/%u (peak %u), skipped%s",
                static_cast<unsigned long long>(published()),
                static_cast<unsigned long long>(drops()), in_use(), slots(), peak_in_use(),
                skips.c_str());
  return line;
}

}  // namespace perception
