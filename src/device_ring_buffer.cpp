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
      ready_(slot_count, nullptr),
      meta_(slot_count),
      generation_(slot_count) {
  if (slot_count == 0 || slot_bytes == 0) {
    throw std::runtime_error("DeviceRingBuffer: empty ring");
  }

  cude_error_check(cudaSetDevice(device_id), "cudaSetDevice");

  try {
    for (uint32_t i = 0; i < slot_count; ++i) {
      cude_error_check(cudaMalloc(&buffers_[i], slot_bytes_), "cudaMalloc");
      cude_error_check(cudaEventCreateWithFlags(&ready_[i], cudaEventDisableTiming), "cudaEventCreateWithFlags");
    }
  } catch (...) {
    release();
    throw;
  }
}

DeviceRingBuffer::~DeviceRingBuffer() { release(); }

void DeviceRingBuffer::release() noexcept {
  for (std::size_t i = 0; i < buffers_.size(); ++i) {
    if (ready_[i]) {
      cudaEventDestroy(ready_[i]);
      ready_[i] = nullptr;
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

  cude_error_check(cudaEventSynchronize(ready_[slot]), "cudaEventSynchronize");

  generation_[slot].fetch_add(1, std::memory_order_release);
  return slot;
}

void* DeviceRingBuffer::data_at_slot(uint32_t slot) { return buffers_[slot]; }

void DeviceRingBuffer::mark_written(uint32_t slot, const FrameMeta& meta, cudaStream_t stream) {
  cude_error_check(cudaEventRecord(ready_[slot], stream), "cudaEventRecord");

  meta_[slot] = meta;
  latest_.store(slot, std::memory_order_release);
}

bool DeviceRingBuffer::fetch_latest_slot(FrameView& out) const {
  const uint32_t slot = latest_.load(std::memory_order_acquire);
  if (slot == kNoSlot) return false;
  
  out.generation = generation_[slot].load(std::memory_order_acquire);

  out.meta = meta_[slot];
  out.ptr = buffers_[slot];
  out.space = MemSpace::Device;
  out.ready = ready_[slot];
  out.slot = slot;
  return true;
}

bool DeviceRingBuffer::still_valid(const FrameView& view) const {
  return generation_[view.slot].load(std::memory_order_acquire) == view.generation;
}

bool DeviceRingBuffer::snapshot_latest(FrameView& out, void* dst, cudaStream_t stream,
                                       cudaEvent_t copied, uint32_t max_attempts) const {
  for (uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
    FrameView view;
    if (!fetch_latest_slot(view)) return false;

    cude_error_check(cudaStreamWaitEvent(stream, view.ready, 0), "cudaStreamWaitEvent");
    cude_error_check(cudaMemcpyAsync(dst, view.ptr, slot_bytes_, cudaMemcpyDeviceToDevice, stream),
                     "cudaMemcpyAsync");
    cude_error_check(cudaEventRecord(copied, stream), "cudaEventRecord");
    cude_error_check(cudaEventSynchronize(copied), "cudaEventSynchronize");

    if (still_valid(view)) {
      out = view;
      out.ptr = dst;
      return true;
    }
  }
  return false;
}

}  // namespace perception
