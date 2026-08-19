#include "ring_lease.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

#include "device_ring_buffer.hpp"

namespace perception {

ReadLease::~ReadLease() {
  if (!ring_) return;
  const cudaError_t err = ring_->drop_slot_hold(slot_, consumer_id_, stream_);
  ring_ = nullptr;
  if (err != cudaSuccess) {
    std::fprintf(stderr, "~ReadLease: cudaEventRecord(read_done) failed: %s\n",
                 cudaGetErrorString(err));
  }
}

ReadLease::ReadLease(ReadLease&& other) noexcept
    : ring_(other.ring_),
      slot_(other.slot_),
      seq_(other.seq_),
      consumer_id_(other.consumer_id_),
      stream_(other.stream_),
      timestamp_ns_(other.timestamp_ns_),
      data_(other.data_),
      data_ready_event_(other.data_ready_event_),
      texture_(other.texture_) {
  other.ring_ = nullptr;
}

ReadLease& ReadLease::operator=(ReadLease&& other) noexcept {
  if (this != &other) {
    if (ring_) {
      const cudaError_t err = ring_->drop_slot_hold(slot_, consumer_id_, stream_);
      if (err != cudaSuccess) {
        std::fprintf(stderr, "ReadLease::operator=: cudaEventRecord(read_done) failed: %s\n",
                     cudaGetErrorString(err));
      }
    }
    ring_ = other.ring_;
    slot_ = other.slot_;
    seq_ = other.seq_;
    consumer_id_ = other.consumer_id_;
    stream_ = other.stream_;
    timestamp_ns_ = other.timestamp_ns_;
    data_ = other.data_;
    data_ready_event_ = other.data_ready_event_;
    texture_ = other.texture_;
    other.ring_ = nullptr;
  }
  return *this;
}

void ReadLease::drop_hold() {
  if (!ring_) return;
  DeviceRingBuffer* ring = ring_;
  ring_ = nullptr;
  const cudaError_t err = ring->drop_slot_hold(slot_, consumer_id_, stream_);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("ReadLease::release: cudaEventRecord(read_done): ") +
                             cudaGetErrorString(err));
  }
}

WriteLease::~WriteLease() {
  if (ring_) ring_->abandon_slot(slot_);
}

WriteLease::WriteLease(WriteLease&& other) noexcept
    : ring_(other.ring_), slot_(other.slot_), stream_(other.stream_), data_(other.data_) {
  other.ring_ = nullptr;
}

WriteLease& WriteLease::operator=(WriteLease&& other) noexcept {
  if (this != &other) {
    if (ring_) ring_->abandon_slot(slot_);
    ring_ = other.ring_;
    slot_ = other.slot_;
    stream_ = other.stream_;
    data_ = other.data_;
    other.ring_ = nullptr;
  }
  return *this;
}

void WriteLease::publish(uint64_t timestamp_ns) {
  if (!ring_) throw std::runtime_error("WriteLease::publish: lease is spent");
  ring_->publish_slot(slot_, timestamp_ns, stream_);
  ring_ = nullptr;
}

}  // namespace perception
