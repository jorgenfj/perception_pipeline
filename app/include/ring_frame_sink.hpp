#pragma once

#include <cstdint>

#include "frame_sink.hpp"
#include "host_frame_ring.hpp"
#include "host_ingress_ring.hpp"
#include "latency_probe.hpp"

namespace perception {

// Binds a camera source to an ingress ring. This is the only place the two
// subprojects meet: spinnaker/ knows FrameSink and nothing about CUDA,
// pipeline/ knows CUDA and nothing about cameras, and the composing app owns
// the adapter that joins them.
class RingFrameSink final : public FrameSink {
 public:
  // `probe` is optional and observation-only. It belongs here because commit()
  // is the earliest moment the host has the frame and the last one still on the
  // acquisition thread -- stamping any later would fold this stage's own
  // queueing into the clock offset it is trying to measure.
  // `epoch_offset_ns` is subtracted from every frame's timestamp, and this is
  // the ONLY place that happens: a PTP camera stamps in TAI, everything from
  // the ingress ring onwards is CLOCK_REALTIME, and this adapter is where the
  // camera's half of the tree meets the pipeline's.
  explicit RingFrameSink(HostIngressRing& ring, LatencyProbe* probe = nullptr,
                         int64_t epoch_offset_ns = 0)
      : ring_(&ring), probe_(probe), epoch_offset_ns_(epoch_offset_ns) {
    if (ring_->fill_mode() != FillMode::External) {
      throw std::runtime_error("RingFrameSink: ring must be FillMode::External");
    }
  }
  void set_tap(HostFrameRing* tap) { tap_ = tap; }

  uint32_t slot_count() const override { return ring_->slot_count(); }
  std::size_t slot_bytes() const override { return ring_->slot_bytes(); }
  void* const* buffers() const override { return ring_->buffers(); }

  uint32_t slot_of(const void* ptr) const override {
    const uint32_t slot = ring_->slot_of(ptr);
    return slot == HostIngressRing::kNoSlot ? kNoSlot : slot;
  }

  // meta.host_recv_ns and meta.frame_id are not forwarded: the ingress ring
  // carries the camera timestamp and nothing else, and the probe wants its own
  // reading of arrival rather than the transport's.
  void commit(uint32_t slot, const FrameMeta& meta) override {
    const uint64_t timestamp_ns = to_host_epoch(meta.timestamp_ns);

    if (probe_) probe_->on_arrival(timestamp_ns);

  
    if (tap_) tap_->publish(ring_->buffers()[slot], meta.bytes, timestamp_ns);

    ring_->commit_external(slot, timestamp_ns, meta.bytes);
  }

  bool consumed(uint32_t slot) override { return ring_->slot_consumed(slot); }

 private:
  uint64_t to_host_epoch(uint64_t camera_ns) const {
    return static_cast<uint64_t>(static_cast<int64_t>(camera_ns) - epoch_offset_ns_);
  }

  HostIngressRing* ring_;
  LatencyProbe* probe_ = nullptr;
  HostFrameRing* tap_ = nullptr;
  int64_t epoch_offset_ns_ = 0;
};

}  // namespace perception
