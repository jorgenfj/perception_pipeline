#include "heap_frame_sink.hpp"

#include <cstdlib>
#include <stdexcept>

namespace perception {
namespace {

// Page-aligned, because a transport DMAing into these does far better with
// whole pages and some drivers require the alignment outright.
constexpr std::size_t kPageSize = 4096;

std::size_t roundUp(std::size_t value, std::size_t to) {
  return ((value + to - 1) / to) * to;
}

}  // namespace

HeapFrameSink::HeapFrameSink(uint32_t slot_count, std::size_t slot_bytes)
    : slot_bytes_(roundUp(slot_bytes, kPageSize)),
      buffers_(slot_count, nullptr),
      meta_(slot_count),
      state_(slot_count, SlotState::Idle) {
  if (slot_count == 0 || slot_bytes == 0) {
    throw std::runtime_error("HeapFrameSink: empty sink");
  }

  for (uint32_t i = 0; i < slot_count; ++i) {
    buffers_[i] = std::aligned_alloc(kPageSize, slot_bytes_);
    if (!buffers_[i]) {
      for (uint32_t j = 0; j < i; ++j) std::free(buffers_[j]);
      throw std::runtime_error("HeapFrameSink: allocation failed");
    }
  }
}

HeapFrameSink::~HeapFrameSink() {
  for (void* buffer : buffers_) std::free(buffer);
}

uint32_t HeapFrameSink::slot_of(const void* ptr) const {
  for (std::size_t i = 0; i < buffers_.size(); ++i) {
    if (buffers_[i] == ptr) return static_cast<uint32_t>(i);
  }
  return kNoSlot;
}

void HeapFrameSink::commit(uint32_t slot, const FrameMeta& meta) {
  if (slot >= slot_count()) throw std::runtime_error("HeapFrameSink::commit: bad slot");
  if (meta.bytes > slot_bytes_) throw std::runtime_error("HeapFrameSink::commit: frame exceeds slot");
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Anything but Idle means the transport refilled a slot still in flight,
    // which it can only do if its handle went back too early.
    if (state_[slot] != SlotState::Idle) {
      throw std::runtime_error("HeapFrameSink::commit: slot is still in flight");
    }
    meta_[slot] = meta;
    state_[slot] = SlotState::InFlight;
    ready_.push_back(slot);
  }
  frame_ready_.notify_one();
}

bool HeapFrameSink::consumed(uint32_t slot) {
  if (slot >= slot_count()) throw std::runtime_error("HeapFrameSink::consumed: bad slot");
  std::lock_guard<std::mutex> lock(mutex_);
  return state_[slot] == SlotState::Idle;
}

bool HeapFrameSink::pop(Frame& out, std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!frame_ready_.wait_for(lock, timeout, [this] { return !running_ || !ready_.empty(); })) {
    return false;
  }
  if (!running_ || ready_.empty()) return false;

  const uint32_t slot = ready_.front();
  ready_.pop_front();

  out.slot = slot;
  out.data = buffers_[slot];
  out.meta = meta_[slot];
  return true;
}

void HeapFrameSink::release(uint32_t slot) {
  if (slot >= slot_count()) throw std::runtime_error("HeapFrameSink::release: bad slot");
  std::lock_guard<std::mutex> lock(mutex_);
  state_[slot] = SlotState::Idle;
}

void HeapFrameSink::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
  }
  frame_ready_.notify_all();
}

}  // namespace perception
