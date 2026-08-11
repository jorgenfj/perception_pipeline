#include "device_ring_buffer.hpp"

#include <stdexcept>
#include <string>

namespace perception {
namespace {

void cude_error_check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

}  // namespace

DeviceRingBuffer::DeviceRingBuffer(uint32_t slot_count, std::size_t slot_bytes, int device_id)
    : slot_bytes_(slot_bytes),
      buffers_(slot_count, nullptr),
      data_ready_event_(slot_count, nullptr),
      seq_(slot_count),
      timestamp_ns_(slot_count) {
  if (slot_count == 0 || slot_bytes == 0) {
    throw std::runtime_error("DeviceRingBuffer: empty ring");
  }

  cude_error_check(cudaSetDevice(device_id), "cudaSetDevice");

  try {
    for (uint32_t i = 0; i < slot_count; ++i) {
      cude_error_check(cudaMalloc(&buffers_[i], slot_bytes_), "cudaMalloc");
      cude_error_check(cudaEventCreateWithFlags(&data_ready_event_[i], cudaEventDisableTiming), "cudaEventCreateWithFlags");
    }
  } catch (...) {
    release();
    throw;
  }
}

DeviceRingBuffer::~DeviceRingBuffer() { release(); }

void DeviceRingBuffer::release() noexcept {
  for (std::size_t i = 0; i < buffers_.size(); ++i) {
    if (data_ready_event_[i]) {
      cudaEventDestroy(data_ready_event_[i]);
      data_ready_event_[i] = nullptr;
    }
    if (buffers_[i]) {
      cudaFree(buffers_[i]);
      buffers_[i] = nullptr;
    }
  }
}

uint32_t DeviceRingBuffer::acquire_write_slot() {
  const uint32_t slot = next_write_slot_;
  next_write_slot_ = (next_write_slot_ + 1) % slot_count();

  cude_error_check(cudaEventSynchronize(data_ready_event_[slot]), "cudaEventSynchronize");

  seq_[slot].fetch_add(1, std::memory_order_release); // -> odd
  std::atomic_thread_fence(std::memory_order_release);
  return slot;
}

void* DeviceRingBuffer::data_at_slot(uint32_t slot) { return buffers_[slot]; }

void DeviceRingBuffer::mark_slot_written(uint32_t slot, uint64_t timestamp_ns, cudaStream_t stream) {
  cude_error_check(cudaEventRecord(data_ready_event_[slot], stream), "cudaEventRecord");

  timestamp_ns_[slot].store(timestamp_ns, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  seq_[slot].fetch_add(1, std::memory_order_relaxed);   // -> even
  latest_.store(slot, std::memory_order_release);
}

bool DeviceRingBuffer::view_latest_inplace(FrameView& out) const {
  const uint32_t slot = latest_.load(std::memory_order_acquire);
  if (slot == kNoSlot) return false;

  uint64_t slot_seq = seq_[slot].load(std::memory_order_acquire);
  if (slot_seq & 1u) return false;

  out.slot_seq = slot_seq;

  out.frame.timestamp_ns = timestamp_ns_[slot];
  out.frame.image_ptr = buffers_[slot];
  out.data_ready_event = data_ready_event_[slot];
  out.slot = slot;

  return true;
}

bool DeviceRingBuffer::get_view_by_timestamp(uint64_t timestamp_ns, uint64_t tol, FrameView& out, bool closest_match) const {
  uint64_t best_diff = UINT64_MAX;
  FrameView best_view;
  bool found = false;

  uint32_t slot_match = 0;
  for (uint32_t slot = 0; slot < slot_count(); ++slot) {
    uint64_t seq = seq_[slot].load(std::memory_order_acquire);
    if (seq == 0) continue;
    if (seq & 1u) continue;
    
    uint64_t slot_timestamp = timestamp_ns_[slot].load(std::memory_order_relaxed);
    uint64_t diff = (timestamp_ns > slot_timestamp) ? (timestamp_ns - slot_timestamp) : (slot_timestamp - timestamp_ns);
    if (diff > tol) continue;

    FrameView cand;
    cand.slot = slot;
    cand.slot_seq = seq;
    cand.data_ready_event = data_ready_event_[slot];
    cand.frame.image_ptr = buffers_[slot];
    cand.frame.timestamp_ns = slot_timestamp;

    // Single atomic per frame view field. No recheck of seq needed.
    if (!closest_match) { out = cand; return true; }

    if (diff < best_diff) { best_diff = diff; best_view = cand; found = true; }
  }

  if (found) { out = best_view; return true; }
  return false;
}

bool DeviceRingBuffer::read_was_clean(const FrameView& view) const {
  return (seq_[view.slot].load(std::memory_order_acquire) == view.slot_seq) && !(view.slot_seq & 1u);
}

bool DeviceRingBuffer::snapshot_latest(FrameView& out, void* dst, cudaStream_t stream,
                                       cudaEvent_t copied, uint32_t max_attempts) const {
  for (uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
    FrameView view;
    if (!view_latest_inplace(view)) continue;

    cude_error_check(cudaStreamWaitEvent(stream, view.data_ready_event, 0), "cudaStreamWaitEvent");
    cude_error_check(cudaMemcpyAsync(dst, view.frame.image_ptr, slot_bytes_, cudaMemcpyDeviceToDevice, stream),
                     "cudaMemcpyAsync");
    cude_error_check(cudaEventRecord(copied, stream), "cudaEventRecord");
    cude_error_check(cudaEventSynchronize(copied), "cudaEventSynchronize");

    if (read_was_clean(view)) {
      out = view;
      out.frame.image_ptr = dst;
      out.data_ready_event = nullptr;
      return true;
    }
  }
  return false;
}

}  // namespace perception
