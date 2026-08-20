#include "upload_stage.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace perception {

namespace {

class StagedHold {
 public:
  StagedHold(HostIngressRing& ring, uint32_t slot, cudaStream_t stream)
      : ring_(&ring), slot_(slot), stream_(stream) {}

  ~StagedHold() {
    if (!ring_) return;
    try {
      ring_->release(slot_, stream_);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "UploadStage: ingress slot %u stranded: %s\n", slot_, e.what());
    }
  }

  StagedHold(const StagedHold&) = delete;
  StagedHold& operator=(const StagedHold&) = delete;

  void release() {
    if (!ring_) return;
    HostIngressRing* ring = ring_;
    ring_ = nullptr;
    ring->release(slot_, stream_);
  }

 private:
  HostIngressRing* ring_;
  uint32_t slot_;
  cudaStream_t stream_;
};

}  // namespace

UploadStage::UploadStage(HostIngressRing& in, DeviceRingBuffer& out, DeviceTransform* transform,
                         const ImageDesc& input_desc, Config config)
    : in_(&in),
      out_(&out),
      transform_(transform),
      input_desc_(input_desc),
      output_desc_(transform ? transform->output_desc(input_desc) : input_desc),
      config_(config),
      scratch_(transform ? config.scratch_slots : 0, nullptr),
      scratch_free_(transform ? config.scratch_slots : 0, nullptr) {
  if (transform_ && config_.scratch_slots == 0) {
    throw std::runtime_error("UploadStage: a transform needs at least one scratch slot");
  }
  if (input_desc_.bytes() == 0 || output_desc_.bytes() == 0) {
    throw std::runtime_error("UploadStage: zero-sized geometry");
  }
  if (in_->slot_bytes() < input_desc_.bytes()) {
    throw std::runtime_error("UploadStage: ingress slots are smaller than one input frame");
  }
  if (out_->slot_bytes() < output_desc_.bytes()) {
    throw std::runtime_error(std::string("UploadStage: output ring slots are smaller than what ") +
                             (transform_ ? transform_->name() : "the upload") + " produces");
  }

  if (config_.use_graph && !transform_) {
    throw std::runtime_error(
        "UploadStage: graph capture needs a transform; with none the captured region would be a "
        "single memcpy");
  }
  if (config_.use_graph && out_->write_policy() != WritePolicy::RoundRobin) {
    throw std::runtime_error(
        "UploadStage: graph capture needs WritePolicy::RoundRobin on the output ring; "
        "ScanForFree picks slots from runtime lease state, so the index a graph bakes is no "
        "longer predictable");
  }
  if (config_.use_graph) {
    const uint32_t out_slots = out_->slot_count();
    if (out_slots % in_->slot_count() != 0) {
      throw std::runtime_error(
          "UploadStage: with use_graph, the ingress ring depth must divide the output ring depth "
          "so the two stay in lockstep across replays");
    }
    if (out_slots % scratch_.size() != 0) {
      throw std::runtime_error(
          "UploadStage: with use_graph, scratch_slots must divide the output ring depth so the "
          "two stay in lockstep across replays");
    }
  }

  cuda_error_check(cudaSetDevice(config_.device_id), "UploadStage: cudaSetDevice");

  try {
    for (std::size_t i = 0; i < scratch_.size(); ++i) {
      cuda_error_check(cudaMalloc(&scratch_[i], input_desc_.bytes()), "UploadStage: cudaMalloc");
      cuda_error_check(cudaEventCreateWithFlags(&scratch_free_[i], cudaEventDisableTiming),
                 "UploadStage: cudaEventCreateWithFlags");
    }
    if (transform_) {
      transform_->prepare(input_desc_, stream_);
      cuda_error_check(cudaStreamSynchronize(stream_), "UploadStage: cudaStreamSynchronize");
    }
    if (config_.use_graph) capture_graphs();
  } catch (...) {
    release();
    throw;
  }
}

void UploadStage::capture_graphs() {
  graph_exec_.assign(out_->slot_count(), nullptr);

  for (uint32_t slot = 0; slot < out_->slot_count(); ++slot) {
    // Divisibility, checked in the constructor, is what makes these the indices
    // this graph will always be replayed with.
    const uint32_t pinned = slot % in_->slot_count();
    const uint32_t scratch = slot % static_cast<uint32_t>(scratch_.size());

    cudaGraph_t graph = nullptr;
    cuda_error_check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
                     "UploadStage: cudaStreamBeginCapture");
    try {
      cuda_error_check(cudaMemcpyAsync(scratch_[scratch], in_->host_ptr(pinned),
                                       input_desc_.bytes(), cudaMemcpyHostToDevice, stream_),
                       "UploadStage: cudaMemcpyAsync(capture)");
      transform_->enqueue(scratch_[scratch], input_desc_, out_->data_at_slot(slot), output_desc_,
                          stream_);
    } catch (...) {
      cudaStreamEndCapture(stream_, &graph);
      if (graph) cudaGraphDestroy(graph);
      throw;
    }
    cuda_error_check(cudaStreamEndCapture(stream_, &graph), "UploadStage: cudaStreamEndCapture");

    const cudaError_t instantiated = cudaGraphInstantiateWithFlags(&graph_exec_[slot], graph, 0);
    cudaGraphDestroy(graph);
    cuda_error_check(instantiated, "UploadStage: cudaGraphInstantiateWithFlags");
  }
}

void UploadStage::enqueue_eager(const HostIngressRing::Staged& staged, uint32_t scratch,
                                void* dst, const std::function<void()>& release_pinned) {
  cuda_error_check(cudaMemcpyAsync(scratch_[scratch], staged.data, staged.bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "UploadStage: cudaMemcpyAsync(H2D)");
  // The pinned slot's only reader is the copy just enqueued above, so it can
  // go back to the ring now rather than waiting for the transform too.
  release_pinned();
  transform_->enqueue(scratch_[scratch], input_desc_, dst, output_desc_, stream_);
}

UploadStage::~UploadStage() {
  stop();
  cudaStreamSynchronize(stream_);
  release();
}

void UploadStage::release() noexcept {
  for (cudaGraphExec_t exec : graph_exec_) {
    if (exec) cudaGraphExecDestroy(exec);
  }
  graph_exec_.clear();

  for (std::size_t i = 0; i < scratch_.size(); ++i) {
    if (scratch_free_[i]) {
      cudaEventDestroy(scratch_free_[i]);
      scratch_free_[i] = nullptr;
    }
    if (scratch_[i]) {
      cudaFree(scratch_[i]);
      scratch_[i] = nullptr;
    }
  }
}

bool UploadStage::step(std::chrono::milliseconds timeout) {
  HostIngressRing::Staged staged;
  if (!in_->pop(staged, timeout)) return false;
  return process(staged);
}

bool UploadStage::step_blocking(std::stop_token stoken) {
  HostIngressRing::Staged staged;
  if (!in_->pop(staged, std::move(stoken))) return false;
  return process(staged);
}

bool UploadStage::process(const HostIngressRing::Staged& staged) {
  StagedHold hold(*in_, staged.slot, stream_);

  if (staged.bytes > input_desc_.bytes()) {
    throw std::runtime_error("UploadStage: staged frame is larger than the declared input");
  }

  if (transform_ == nullptr) {
    WriteLease out_lease = out_->acquire_write(stream_);
    cuda_error_check(cudaMemcpyAsync(out_lease.data(), staged.data, staged.bytes,
                               cudaMemcpyHostToDevice, stream_),
               "UploadStage: cudaMemcpyAsync(H2D)");
    hold.release();
    out_lease.publish(staged.timestamp_ns);

    uploaded_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  WriteLease out_lease = out_->acquire_write(stream_);
  const uint32_t slot = out_lease.slot();

  const uint32_t scratch = slot % static_cast<uint32_t>(scratch_.size());
  cuda_error_check(cudaEventSynchronize(scratch_free_[scratch]),
             "UploadStage: cudaEventSynchronize(scratch)");

  if (graph_exec_.empty()) {
    enqueue_eager(staged, scratch, out_lease.data(), [&hold] { hold.release(); });
  } else if (staged.bytes != input_desc_.bytes() ||
             staged.slot != slot % in_->slot_count()) {
    // Log the error, and fallback to eager path
    graph_fallbacks_.fetch_add(1, std::memory_order_relaxed);
    enqueue_eager(staged, scratch, out_lease.data(), [&hold] { hold.release(); });
  } else {
    cuda_error_check(cudaGraphLaunch(graph_exec_[slot], stream_), "UploadStage: cudaGraphLaunch");
    // Hold is on the same stream as the graph,
    // so hold.release event is queued after the graph launch.
    hold.release();
  }

  out_lease.publish(staged.timestamp_ns);
  cuda_error_check(cudaEventRecord(scratch_free_[scratch], stream_),
             "UploadStage: cudaEventRecord(scratch)");

  uploaded_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void UploadStage::start() {
  if (running_.exchange(true)) return;
  worker_ = std::jthread([this](std::stop_token stoken) { run(std::move(stoken)); });
}

void UploadStage::stop() {
  running_.store(false, std::memory_order_relaxed);
  worker_.request_stop();
  if (worker_.joinable()) worker_.join();
}

void UploadStage::run(std::stop_token stoken) {
  try {
    cuda_error_check(cudaSetDevice(config_.device_id), "UploadStage: cudaSetDevice");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "upload stage stopped: %s\n", e.what());
    running_.store(false, std::memory_order_relaxed);
    return;
  }

  while (!stoken.stop_requested() && running_.load(std::memory_order_relaxed)) {
    try {
      if (!step_blocking(stoken) && !in_->running()) break;
    } catch (const std::exception& e) {
      failed_.fetch_add(1, std::memory_order_relaxed);
      std::fprintf(stderr, "upload stage: %s\n", e.what());
    }
  }

  running_.store(false, std::memory_order_relaxed);
}

}  // namespace perception
