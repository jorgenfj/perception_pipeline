#include "transform_stage.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace perception {

TransformStage::TransformStage(DeviceRingBuffer& in, DeviceRingBuffer& out,
                               DeviceTransform& transform, const ImageDesc& input_desc,
                               Config config)
    : in_(&in),
      out_(&out),
      transform_(&transform),
      input_desc_(input_desc),
      output_desc_(transform.output_desc(input_desc)),
      config_(config) {
  if (in_ == out_) {
    throw std::runtime_error(
        "TransformStage: input and output ring must differ; a transform reads its source while "
        "writing its destination");
  }
  if (input_desc_.bytes() == 0 || output_desc_.bytes() == 0) {
    throw std::runtime_error("TransformStage: zero-sized geometry");
  }
  if (in_->slot_bytes() < input_desc_.bytes()) {
    throw std::runtime_error("TransformStage: input ring slots are smaller than one input frame");
  }
  if (out_->slot_bytes() < output_desc_.bytes()) {
    throw std::runtime_error(std::string("TransformStage: output ring slots are smaller than what ") +
                             transform_->name() + " produces");
  }

  cuda_error_check(cudaSetDevice(config_.device_id), "TransformStage: cudaSetDevice");

  transform_->prepare(input_desc_, stream_);
  cuda_error_check(cudaStreamSynchronize(stream_), "TransformStage: cudaStreamSynchronize");
}

TransformStage::~TransformStage() {
  stop();
  cudaStreamSynchronize(stream_);
}

bool TransformStage::step() {
  // Unleased peek first, so an idle poll costs a couple of atomic loads rather
  // than a lease acquire and release.
  FrameView peek;
  if (!in_->view_latest_inplace(peek)) return false;
  if (have_last_ && peek.slot == last_slot_ && peek.slot_seq == last_seq_) return false;

  // The lease pins whatever discovery lands on -- possibly newer than the peek,
  // which is fine, latest-wins either way. From here the source cannot be
  // recycled underneath the transform, which is what makes the read safe rather
  // than merely unlikely to be torn.
  ReadLease lease = in_->lease_latest(config_.consumer_id, stream_);
  if (!lease.valid()) {
    missed_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  cuda_error_check(cudaStreamWaitEvent(stream_, lease.data_ready_event(), 0),
                   "TransformStage: cudaStreamWaitEvent");

  WriteLease out_lease = out_->acquire_write(stream_);
  transform_->enqueue(lease.data(), input_desc_, out_lease.data(), output_desc_, stream_);

  // The input frame's capture time travels with the derived frame -- consumers
  // downstream time detections by capture instant, not by arrival.
  out_lease.publish(lease.timestamp_ns());

  // Records read completion against our stream, so the upstream producer orders
  // its next write to this slot behind the transform. Safe to do now: the work
  // is enqueued, and nothing waits for it to finish.
  const uint32_t slot = lease.slot();
  const uint64_t seq = lease.seq();
  lease.drop_hold();

  last_slot_ = slot;
  last_seq_ = seq;
  have_last_ = true;

  transformed_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void TransformStage::start() {
  if (running_.exchange(true)) return;
  worker_ = std::thread(&TransformStage::run, this);
}

void TransformStage::stop() {
  if (!running_.exchange(false)) return;
  if (worker_.joinable()) worker_.join();
}

void TransformStage::run() {
  try {
    cuda_error_check(cudaSetDevice(config_.device_id), "TransformStage: cudaSetDevice");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "transform stage stopped: %s\n", e.what());
    running_.store(false, std::memory_order_relaxed);
    return;
  }

  while (running_.load(std::memory_order_relaxed)) {
    try {
      if (!step()) std::this_thread::sleep_for(config_.poll_interval);
    } catch (const std::exception& e) {
      failed_.fetch_add(1, std::memory_order_relaxed);
      std::fprintf(stderr, "transform stage: %s\n", e.what());
    }
  }
}

}  // namespace perception
