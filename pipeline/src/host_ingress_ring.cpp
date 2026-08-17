#include "host_ingress_ring.hpp"

#include <stdexcept>
#include <string>

#include "cuda_util.hpp"

namespace perception {

HostIngressRing::HostIngressRing(uint32_t slot_count, std::size_t slot_bytes, int device_id,
                                 FillMode fill)
    : slot_bytes_(slot_bytes),
      fill_(fill),
      host_(slot_count, nullptr),
      h2d_done_(slot_count, nullptr),
      timestamp_ns_(slot_count, 0),
      bytes_(slot_count, 0),
      state_(slot_count, SlotState::Idle) {
  if (slot_count == 0 || slot_bytes == 0) {
    throw std::runtime_error("HostIngressRing: empty ring");
  }

  cuda_error_check(cudaSetDevice(device_id), "HostIngressRing: cudaSetDevice");

  try {
    for (uint32_t i = 0; i < slot_count; ++i) {
      cuda_error_check(cudaMallocHost(&host_[i], slot_bytes_), "HostIngressRing: cudaMallocHost");
      cuda_error_check(cudaEventCreateWithFlags(&h2d_done_[i], cudaEventDisableTiming),
                 "HostIngressRing: cudaEventCreateWithFlags");
    }
  } catch (...) {
    release_all();
    throw;
  }
}

HostIngressRing::~HostIngressRing() {
  stop();
  release_all();
}

void HostIngressRing::release_all() noexcept {
  for (std::size_t i = 0; i < host_.size(); ++i) {
    if (h2d_done_[i]) {
      cudaEventSynchronize(h2d_done_[i]);
      cudaEventDestroy(h2d_done_[i]);
      h2d_done_[i] = nullptr;
    }
    if (host_[i]) {
      cudaFreeHost(host_[i]);
      host_[i] = nullptr;
    }
  }
}

uint32_t HostIngressRing::acquire() {
  if (fill_ != FillMode::Staged) {
    throw std::runtime_error("HostIngressRing::acquire: ring is External; the transport picks");
  }
  uint32_t slot;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    slot_free_.wait(lock, [this] {
      return !running_ || (acquired_ - released_) < slot_count();
    });
    if (!running_) return kNoSlot;

    slot = static_cast<uint32_t>(acquired_ % slot_count());
    ++acquired_;
  }

  cuda_error_check(cudaEventSynchronize(h2d_done_[slot]), "HostIngressRing: cudaEventSynchronize");
  return slot;
}

void* HostIngressRing::host_ptr(uint32_t slot) {
  if (slot >= slot_count()) throw std::runtime_error("HostIngressRing::host_ptr: bad slot");
  return host_[slot];
}

void HostIngressRing::commit(uint32_t slot, uint64_t timestamp_ns, std::size_t bytes) {
  if (fill_ != FillMode::Staged) {
    throw std::runtime_error("HostIngressRing::commit: ring is External; use commit_external");
  }
  if (slot >= slot_count()) throw std::runtime_error("HostIngressRing::commit: bad slot");
  if (bytes > slot_bytes_) {
    throw std::runtime_error("HostIngressRing::commit: frame larger than the staging slot");
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // acquired_ > committed_ is the "something is outstanding" test, and it is
    // what rejects a double commit -- the slot index alone still matches on the
    // second call, which would hand the consumer a slot never refilled.
    if (acquired_ == committed_ ||
        slot != static_cast<uint32_t>((acquired_ - 1) % slot_count())) {
      throw std::runtime_error("HostIngressRing::commit: not the slot most recently acquired");
    }
    timestamp_ns_[slot] = timestamp_ns;
    bytes_[slot] = bytes;
    state_[slot] = SlotState::Committed;
    ready_.push_back(slot);
    ++committed_;
  }
  frame_ready_.notify_one();
}

void HostIngressRing::abandon(uint32_t slot) {
  if (fill_ != FillMode::Staged) {
    throw std::runtime_error("HostIngressRing::abandon: ring is External; nothing was acquired");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (acquired_ == committed_ || slot != static_cast<uint32_t>((acquired_ - 1) % slot_count())) {
    throw std::runtime_error("HostIngressRing::abandon: not the slot most recently acquired");
  }
  --acquired_;
}

uint32_t HostIngressRing::slot_of(const void* ptr) const {
  for (std::size_t i = 0; i < host_.size(); ++i) {
    if (host_[i] == ptr) return static_cast<uint32_t>(i);
  }
  return kNoSlot;
}

void HostIngressRing::commit_external(uint32_t slot, uint64_t timestamp_ns, std::size_t bytes) {
  if (fill_ != FillMode::External) {
    throw std::runtime_error("HostIngressRing::commit_external: ring is Staged; use commit");
  }
  if (slot >= slot_count()) throw std::runtime_error("HostIngressRing::commit_external: bad slot");
  if (bytes > slot_bytes_) {
    throw std::runtime_error("HostIngressRing::commit_external: frame larger than the slot");
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Anything but Idle means the transport refilled a slot that is still in
    // flight, which it can only do if its handle was handed back too early --
    // the frame the consumer is holding has already been overwritten.
    if (state_[slot] != SlotState::Idle) {
      throw std::runtime_error("HostIngressRing::commit_external: slot is still in flight");
    }
    timestamp_ns_[slot] = timestamp_ns;
    bytes_[slot] = bytes;
    state_[slot] = SlotState::Committed;
    ready_.push_back(slot);
    ++committed_;
  }
  frame_ready_.notify_one();
}

bool HostIngressRing::slot_consumed(uint32_t slot) {
  if (slot >= slot_count()) {
    throw std::runtime_error("HostIngressRing::slot_consumed: bad slot");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  // Only Releasing means h2d_done_ describes this frame. A never-recorded event
  // queries as complete, so asking earlier would report a slot free while the
  // consumer still has it.
  if (state_[slot] != SlotState::Releasing) return false;
  if (cudaEventQuery(h2d_done_[slot]) != cudaSuccess) return false;
  state_[slot] = SlotState::Idle;
  return true;
}

bool HostIngressRing::pop(Staged& out, std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!frame_ready_.wait_for(lock, timeout, [this] { return !running_ || !ready_.empty(); })) {
    return false;
  }
  if (!running_ || ready_.empty()) return false;

  const uint32_t slot = ready_.front();
  ready_.pop_front();

  out.slot = slot;
  out.data = host_[slot];
  out.bytes = bytes_[slot];
  out.timestamp_ns = timestamp_ns_[slot];
  return true;
}

void HostIngressRing::release(uint32_t slot, cudaStream_t stream) {
  if (slot >= slot_count()) throw std::runtime_error("HostIngressRing::release: bad slot");
  reject_default_stream(stream, "HostIngressRing::release");

  // Record before publishing the release: the producer must never observe the
  // slot as free while the event still describes an older upload.
  cuda_error_check(cudaEventRecord(h2d_done_[slot], stream), "HostIngressRing: cudaEventRecord");
  {
    std::lock_guard<std::mutex> lock(mutex_);
    state_[slot] = SlotState::Releasing;
    ++released_;
  }
  slot_free_.notify_one();
}

void HostIngressRing::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) return;
    running_ = false;
  }
  slot_free_.notify_all();
  frame_ready_.notify_all();
}

bool HostIngressRing::running() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return running_;
}

uint64_t HostIngressRing::committed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return committed_;
}

uint64_t HostIngressRing::consumed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return released_;
}

}  // namespace perception
