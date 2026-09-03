#include "host_frame_pool.hpp"

#include <stdexcept>

namespace perception {

HostFramePool::Store::~Store() {
  // Every frame is back by construction: this runs when the last shared_ptr to
  // the Store goes, and each outstanding frame holds one.
  if (block) cudaFreeHost(block);
}

void HostFramePool::Store::release(uint32_t slot) noexcept {
  {
    const std::lock_guard<std::mutex> lock(mutex);
    free.push_back(slot);
  }
  in_use.fetch_sub(1, std::memory_order_relaxed);
}

HostFramePool::HostFramePool(uint32_t slots, std::size_t frame_bytes, uint32_t width,
                             uint32_t height, int device_id)
    : store_(std::make_shared<Store>()) {
  if (slots == 0) throw std::runtime_error("HostFramePool: needs at least one slot");
  if (frame_bytes == 0) throw std::runtime_error("HostFramePool: frame_bytes is zero");

  store_->frame_bytes = frame_bytes;
  store_->device_id = device_id;

  cuda_error_check(cudaSetDevice(device_id), "HostFramePool: cudaSetDevice");
  // One allocation for the lot. cudaMallocHost pins pages with the kernel and
  // is expensive enough that doing it per slot is measurable at startup and
  // unthinkable per frame.
  cuda_error_check(cudaMallocHost(&store_->block, frame_bytes * slots),
                   "HostFramePool: cudaMallocHost");

  store_->frames.resize(slots);
  store_->free.reserve(slots);
  for (uint32_t i = 0; i < slots; ++i) {
    HostFrame& frame = store_->frames[i];
    frame.data = static_cast<unsigned char*>(store_->block) + static_cast<std::size_t>(i) * frame_bytes;
    frame.bytes = frame_bytes;
    frame.width = width;
    frame.height = height;
    // Handed out newest-first below, which keeps a lightly loaded pool reusing
    // the same few slots and their pages hot.
    store_->free.push_back(slots - 1 - i);
  }
}

std::shared_ptr<HostFrame> HostFramePool::acquire(uint64_t timestamp_ns) {
  uint32_t slot = 0;
  {
    const std::lock_guard<std::mutex> lock(store_->mutex);
    if (store_->free.empty()) {
      store_->drops.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    slot = store_->free.back();
    store_->free.pop_back();
  }

  const uint32_t held = store_->in_use.fetch_add(1, std::memory_order_relaxed) + 1;
  uint32_t peak = store_->peak_in_use.load(std::memory_order_relaxed);
  while (held > peak &&
         !store_->peak_in_use.compare_exchange_weak(peak, held, std::memory_order_relaxed)) {
  }
  store_->acquired.fetch_add(1, std::memory_order_relaxed);

  HostFrame& frame = store_->frames[slot];
  frame.timestamp_ns = timestamp_ns;
  frame.host_ready_ns = 0;  // set by the producer when the copy has retired
  frame.sequence = store_->sequence.fetch_add(1, std::memory_order_relaxed);

  // The deleter holds the Store, not the pool: see the comment on Store for why
  // a frame is allowed to outlive the pool that issued it.
  return std::shared_ptr<HostFrame>(&frame, [store = store_, slot](HostFrame*) noexcept {
    store->release(slot);
  });
}

}  // namespace perception
