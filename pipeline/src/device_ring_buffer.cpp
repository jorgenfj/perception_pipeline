#include "device_ring_buffer.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

#include "cuda_util.hpp"

namespace perception {

DeviceRingBuffer::DeviceRingBuffer(uint32_t slot_count, std::size_t slot_bytes,
                                   ReuseWait reuse_wait,
                                   WritePolicy write_policy,
                                   uint32_t max_consumers, int device_id)
    : reuse_wait_(reuse_wait), write_policy_(write_policy),
      max_consumers_(max_consumers), slot_bytes_(slot_bytes),
      buffers_(slot_count, nullptr), data_ready_event_(slot_count, nullptr),
      read_done_event_(static_cast<std::size_t>(slot_count) * max_consumers,
                       nullptr),
      seq_(slot_count), timestamp_ns_(slot_count), readers_(slot_count) {
  if (slot_count < 2 || slot_bytes == 0) {
    throw std::runtime_error(
        "DeviceRingBuffer: needs at least two non-empty slots");
  }
  if (max_consumers == 0) {
    throw std::runtime_error(
        "DeviceRingBuffer: needs room for at least one consumer");
  }

  cuda_error_check(cudaSetDevice(device_id), "cudaSetDevice");

  try {
    for (uint32_t i = 0; i < slot_count; ++i) {
      cuda_error_check(cudaMalloc(&buffers_[i], slot_bytes_), "cudaMalloc");
      cuda_error_check(cudaEventCreateWithFlags(&data_ready_event_[i],
                                                cudaEventDisableTiming),
                       "cudaEventCreateWithFlags");
      for (uint32_t c = 0; c < max_consumers_; ++c) {
        cuda_error_check(
            cudaEventCreateWithFlags(&read_done_event_[i * max_consumers_ + c],
                                     cudaEventDisableTiming),
            "cudaEventCreateWithFlags");
      }
    }
  } catch (...) {
    cleanup();
    throw;
  }
}

DeviceRingBuffer::~DeviceRingBuffer() { cleanup(); }

void DeviceRingBuffer::cleanup() noexcept {
  for (std::size_t i = 0; i < read_done_event_.size(); ++i) {
    if (read_done_event_[i]) {
      cudaEventDestroy(read_done_event_[i]);
      read_done_event_[i] = nullptr;
    }
  }
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

void DeviceRingBuffer::wait_unleased(uint32_t slot) {
  uint32_t held = readers_[slot].load(std::memory_order_seq_cst);
  if (held == 0)
    return;

  write_stalls_.fetch_add(1, std::memory_order_relaxed);
  while (held != 0) {
    // Block if atmoics value equals held
    readers_[slot].wait(held, std::memory_order_acquire);
    held = readers_[slot].load(std::memory_order_acquire);
  }
}

uint32_t DeviceRingBuffer::claim_free_slot() {
  const uint32_t count = slot_count();
  for (;;) {
    const uint32_t newest = latest_.load(std::memory_order_acquire);
    uint32_t busy = kNoSlot;

    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t slot = (next_write_slot_ + i) % count;
      if (slot == newest)
        continue;
      if (readers_[slot].load(std::memory_order_acquire) != 0) {
        if (busy == kNoSlot)
          busy = slot;
        continue;
      }
      next_write_slot_ = (slot + 1) % count;
      return slot;
    }

    // Everything is either leased or the current latest. Park on one of the
    // leased slots and rescan when it changes; any release wakes us.
    if (busy == kNoSlot) {
      throw std::runtime_error(
          "DeviceRingBuffer: no writable slot; ring depth is too small");
    }
    write_stalls_.fetch_add(1, std::memory_order_relaxed);
    readers_[busy].wait(readers_[busy].load(std::memory_order_acquire),
                        std::memory_order_acquire);
  }
}

void DeviceRingBuffer::order_behind_readers(uint32_t slot,
                                            cudaStream_t stream) {
  for (uint32_t c = 0; c < max_consumers_; ++c) {
    cuda_error_check(
        cudaStreamWaitEvent(stream, read_done_event_[slot * max_consumers_ + c],
                            0),
        "cudaStreamWaitEvent(read_done)");
  }
}

WriteLease DeviceRingBuffer::acquire_write(cudaStream_t stream) {
  reject_default_stream(stream, "DeviceRingBuffer::acquire_write");

  uint32_t slot;
  if (write_policy_ == WritePolicy::RoundRobin) {
    slot = next_write_slot_;
    next_write_slot_ = (next_write_slot_ + 1) % slot_count();
  } else {
    slot = claim_free_slot();
  }

  seq_[slot].fetch_add(1, std::memory_order_seq_cst); // -> odd
  wait_unleased(slot);

  order_behind_readers(slot, stream);

  if (reuse_wait_ == ReuseWait::HostSync) {
    cuda_error_check(cudaEventSynchronize(data_ready_event_[slot]),
                     "cudaEventSynchronize");
  } else {
    cuda_error_check(cudaStreamWaitEvent(stream, data_ready_event_[slot], 0),
                     "cudaStreamWaitEvent");
  }

  return WriteLease(this, slot, stream, buffers_[slot]);
}

void DeviceRingBuffer::publish_slot(uint32_t slot, uint64_t timestamp_ns,
                                    cudaStream_t stream) {
  cuda_error_check(cudaEventRecord(data_ready_event_[slot], stream),
                   "cudaEventRecord");

  timestamp_ns_[slot].store(timestamp_ns, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  seq_[slot].fetch_add(1, std::memory_order_relaxed); // -> even
  latest_.store(slot, std::memory_order_release);
}

void DeviceRingBuffer::abandon_slot(uint32_t slot) noexcept {
  timestamp_ns_[slot].store(kNoTimestamp, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  seq_[slot].fetch_add(1, std::memory_order_relaxed); // -> even
}

cudaError_t DeviceRingBuffer::drop_slot_hold(uint32_t slot,
                                             uint32_t consumer_id,
                                             cudaStream_t stream) noexcept {
  const cudaError_t err = cudaEventRecord(
      read_done_event_[slot * max_consumers_ + consumer_id], stream);

  readers_[slot].fetch_sub(1, std::memory_order_acq_rel);
  readers_[slot].notify_all();
  return err;
}

ReadLease DeviceRingBuffer::lease_latest(uint32_t consumer_id,
                                         cudaStream_t stream,
                                         uint32_t max_attempts) {
  if (consumer_id >= max_consumers_) {
    throw std::runtime_error(
        "DeviceRingBuffer::lease_latest: consumer id out of range");
  }
  reject_default_stream(stream, "DeviceRingBuffer::lease_latest");

  for (uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
    const uint32_t slot = latest_.load(std::memory_order_acquire);
    if (slot == kNoSlot)
      return {};

    const uint64_t seq = seq_[slot].load(std::memory_order_acquire);
    if (seq & 1u)
      continue; // producer holds it

    readers_[slot].fetch_add(1, std::memory_order_seq_cst);

    if (seq_[slot].load(std::memory_order_seq_cst) != seq) {
      readers_[slot].fetch_sub(1, std::memory_order_acq_rel);
      readers_[slot].notify_all();
      continue;
    }

    return ReadLease(this, slot, seq, consumer_id, stream,
                     timestamp_ns_[slot].load(std::memory_order_relaxed),
                     buffers_[slot], data_ready_event_[slot]);
  }
  return {};
}

ReadLease DeviceRingBuffer::lease_by_timestamp(
    uint64_t timestamp_ns, uint64_t tol, uint32_t consumer_id,
    cudaStream_t stream, bool closest_match, uint32_t max_attempts) {
  if (consumer_id >= max_consumers_) {
    throw std::runtime_error(
        "DeviceRingBuffer::lease_by_timestamp: consumer id out of range");
  }
  reject_default_stream(stream, "DeviceRingBuffer::lease_by_timestamp");

  for (uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
    uint64_t best_diff = UINT64_MAX;
    uint32_t best_slot = kNoSlot;
    uint64_t best_seq = 0;

    for (uint32_t slot = 0; slot < slot_count(); ++slot) {
      const uint64_t seq = seq_[slot].load(std::memory_order_acquire);
      if (seq == 0 || (seq & 1u))
        continue;

      const uint64_t stamp =
          timestamp_ns_[slot].load(std::memory_order_relaxed);
      if (stamp == kNoTimestamp)
        continue; // abandoned slot

      const uint64_t diff = (timestamp_ns > stamp) ? (timestamp_ns - stamp)
                                                   : (stamp - timestamp_ns);
      if (diff > tol)
        continue;

      if (!closest_match) {
        best_slot = slot;
        best_seq = seq;
        break;
      }
      if (diff < best_diff) {
        best_diff = diff;
        best_slot = slot;
        best_seq = seq;
      }
    }

    if (best_slot == kNoSlot)
      return {};

    readers_[best_slot].fetch_add(1, std::memory_order_seq_cst);
    if (seq_[best_slot].load(std::memory_order_seq_cst) != best_seq) {
      readers_[best_slot].fetch_sub(1, std::memory_order_acq_rel);
      readers_[best_slot].notify_all();
      continue;
    }

    return ReadLease(this, best_slot, best_seq, consumer_id, stream,
                     timestamp_ns_[best_slot].load(std::memory_order_relaxed),
                     buffers_[best_slot], data_ready_event_[best_slot]);
  }
  return {};
}

bool DeviceRingBuffer::get_view_by_timestamp(uint64_t timestamp_ns,
                                             uint64_t tol, FramePeek &out,
                                             bool closest_match) const {
  uint64_t best_diff = UINT64_MAX;
  FramePeek best_view;
  bool found = false;

  for (uint32_t slot = 0; slot < slot_count(); ++slot) {
    const uint64_t seq = seq_[slot].load(std::memory_order_acquire);
    if (seq == 0 || (seq & 1u))
      continue;

    const uint64_t slot_timestamp =
        timestamp_ns_[slot].load(std::memory_order_relaxed);
    if (slot_timestamp == kNoTimestamp)
      continue;

    const uint64_t diff = (timestamp_ns > slot_timestamp)
                              ? (timestamp_ns - slot_timestamp)
                              : (slot_timestamp - timestamp_ns);
    if (diff > tol)
      continue;

    FramePeek cand;
    cand.timestamp_ns = slot_timestamp;
    cand.slot = slot;
    cand.slot_seq = seq;

    if (!closest_match) {
      out = cand;
      return true;
    }
    if (diff < best_diff) {
      best_diff = diff;
      best_view = cand;
      found = true;
    }
  }

  if (found) {
    out = best_view;
    return true;
  }
  return false;
}

bool DeviceRingBuffer::view_latest_inplace(FramePeek &out) const {
  const uint32_t slot = latest_.load(std::memory_order_acquire);
  if (slot == kNoSlot)
    return false;

  const uint64_t slot_seq = seq_[slot].load(std::memory_order_acquire);
  if (slot_seq & 1u)
    return false;

  out.slot_seq = slot_seq;
  out.timestamp_ns = timestamp_ns_[slot].load(std::memory_order_relaxed);
  out.slot = slot;
  return true;
}

bool DeviceRingBuffer::read_was_clean(const FramePeek &view) const {
  return (seq_[view.slot].load(std::memory_order_acquire) == view.slot_seq) &&
         !(view.slot_seq & 1u);
}

} // namespace perception
