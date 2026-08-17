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
      // One-shot transform setup, then drain it: enqueue() is entitled to
      // assume whatever prepare() built is live on the device.
      transform_->prepare(input_desc_, stream_);
      cuda_error_check(cudaStreamSynchronize(stream_), "UploadStage: cudaStreamSynchronize");
    }
    // After prepare(), so whatever it built is live and is not itself captured.
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
    // ThreadLocal, not Global: the producer thread may be inside
    // HostIngressRing::acquire()'s cudaEventSynchronize right now, and Global
    // mode would flag that unrelated call as unsafe.
    cuda_error_check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
                     "UploadStage: cudaStreamBeginCapture");
    try {
      // Capture the device work only. Every event record and wait stays outside
      // the graph, on the stream around the launch -- it keeps the ring APIs
      // untouched and sidesteps the semantics of an event both recorded inside
      // a graph and waited on by a later replay of it.
      cuda_error_check(cudaMemcpyAsync(scratch_[scratch], in_->host_ptr(pinned),
                                       input_desc_.bytes(), cudaMemcpyHostToDevice, stream_),
                       "UploadStage: cudaMemcpyAsync(capture)");
      transform_->enqueue(scratch_[scratch], input_desc_, out_->data_at_slot(slot), output_desc_,
                          stream_);
    } catch (...) {
      // The stream stays in capture mode until it is ended, so it has to be
      // closed even on the failure path or every later launch errors out.
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
                                void* dst) {
  cuda_error_check(cudaMemcpyAsync(scratch_[scratch], staged.data, staged.bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "UploadStage: cudaMemcpyAsync(H2D)");
  transform_->enqueue(scratch_[scratch], input_desc_, dst, output_desc_, stream_);
}

UploadStage::~UploadStage() {
  stop();
  // Nothing may still be reading the scratch buffers when they are freed.
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

bool UploadStage::step() {
  HostIngressRing::Staged staged;
  if (!in_->pop(staged, config_.pop_timeout)) return false;

  StagedHold hold(*in_, staged.slot, stream_);

  if (staged.bytes > input_desc_.bytes()) {
    throw std::runtime_error("UploadStage: staged frame is larger than the declared input");
  }

  if (transform_ == nullptr) {
    // Upload only. The output slot is the copy's destination, so the lease has
    // to come first; the H2D therefore sits inside the lease's window. If
    // anything throws before publish, the lease's destructor restores the slot
    // rather than stranding it.
    WriteLease out_lease = out_->acquire_write(stream_);
    cuda_error_check(cudaMemcpyAsync(out_lease.data(), staged.data, staged.bytes,
                               cudaMemcpyHostToDevice, stream_),
               "UploadStage: cudaMemcpyAsync(H2D)");
    hold.release();
    out_lease.publish(staged.timestamp_ns);

    uploaded_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  // Held from here until publish. A throw in between is no longer a stranded
  // slot -- the lease's destructor restores parity and stamps a sentinel
  // timestamp, which is the abandon path the raw slot API never had.
  WriteLease out_lease = out_->acquire_write(stream_);
  const uint32_t slot = out_lease.slot();

  // Derived from the slot, not counted alongside it: a separate counter went
  // permanently out of phase the first time anything threw before acquire_write,
  // which silently disabled graph replay from then on.
  const uint32_t scratch = slot % static_cast<uint32_t>(scratch_.size());

  // Reuse interlock for the scratch buffer, mirroring the one the two rings
  // run: the transform that last read it must have finished. A never-recorded
  // event counts as complete, so the first lap passes straight through.
  cuda_error_check(cudaEventSynchronize(scratch_free_[scratch]),
             "UploadStage: cudaEventSynchronize(scratch)");

  if (graph_exec_.empty()) {
    enqueue_eager(staged, scratch, out_lease.data());
  } else if (staged.bytes != input_desc_.bytes() ||
             staged.slot != slot % in_->slot_count()) {
    // A captured graph bakes its pointers and its copy size, so it is only
    // valid for the indices it was captured with. The ingress slot advances on
    // its own, so a drift shows up here instead of silently replaying a graph
    // that reads the wrong buffer.
    graph_fallbacks_.fetch_add(1, std::memory_order_relaxed);
    enqueue_eager(staged, scratch, out_lease.data());
  } else {
    cuda_error_check(cudaGraphLaunch(graph_exec_[slot], stream_), "UploadStage: cudaGraphLaunch");
  }

  // Later than the eager path used to release it: the graph is one opaque unit,
  // so the pinned slot comes back after the transform rather than between it and
  // the copy. Stream ordering had already pushed the effective hold out to full
  // stage latency, so this costs nothing real.
  hold.release();

  // Both events sit after the transform on the same stream, so each fires when
  // the buffer it guards stops being read: the output slot for consumers, the
  // scratch slot for the next lap.
  out_lease.publish(staged.timestamp_ns);
  cuda_error_check(cudaEventRecord(scratch_free_[scratch], stream_),
             "UploadStage: cudaEventRecord(scratch)");

  uploaded_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void UploadStage::start() {
  if (running_.exchange(true)) return;
  worker_ = std::thread(&UploadStage::run, this);
}

void UploadStage::stop() {
  if (!running_.exchange(false)) return;
  if (worker_.joinable()) worker_.join();
}

void UploadStage::run() {
  // Per-thread state, so the worker has to set it for itself.
  try {
    cuda_error_check(cudaSetDevice(config_.device_id), "UploadStage: cudaSetDevice");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "upload stage stopped: %s\n", e.what());
    running_.store(false, std::memory_order_relaxed);
    return;
  }

  while (running_.load(std::memory_order_relaxed)) {
    try {
      step();
    } catch (const std::exception& e) {
      // One bad frame should not take the pipeline down, but a persistent
      // fault would otherwise spin silently, so it is counted as well as
      // printed.
      failed_.fetch_add(1, std::memory_order_relaxed);
      std::fprintf(stderr, "upload stage: %s\n", e.what());
    }
  }
}

}  // namespace perception
