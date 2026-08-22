#pragma once

#include "frame_sink.hpp"
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
  explicit RingFrameSink(HostIngressRing& ring, LatencyProbe* probe = nullptr)
      : ring_(&ring), probe_(probe) {
    if (ring_->fill_mode() != FillMode::External) {
      throw std::runtime_error("RingFrameSink: ring must be FillMode::External");
    }
  }

  uint32_t slot_count() const override { return ring_->slot_count(); }
  std::size_t slot_bytes() const override { return ring_->slot_bytes(); }
  void* const* buffers() const override { return ring_->buffers(); }

  uint32_t slot_of(const void* ptr) const override {
    const uint32_t slot = ring_->slot_of(ptr);
    return slot == HostIngressRing::kNoSlot ? kNoSlot : slot;
  }

  // meta.host_recv_ns and meta.frame_id are not forwarded: the ingress ring
  // carries the camera timestamp and nothing else, and the probe wants its own
  // reading of arrival rather than the transport's. They are there for the
  // recorder, which taps the frame before this adapter sees it.
  void commit(uint32_t slot, const FrameMeta& meta) override {
    if (probe_) probe_->on_arrival(meta.timestamp_ns);
    ring_->commit_external(slot, meta.timestamp_ns, meta.bytes);
  }

  bool consumed(uint32_t slot) override { return ring_->slot_consumed(slot); }

 private:
  HostIngressRing* ring_;
  LatencyProbe* probe_ = nullptr;
};

}  // namespace perception
