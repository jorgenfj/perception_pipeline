#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "cuda_util.hpp"
#include "device_ring_buffer.hpp"
#include "device_transform.hpp"
#include "host_ingress_ring.hpp"
#include "types.hpp"

namespace perception {

// The host-to-device boundary: pinned ingress ring -> H2D -> device ring.
// Vendor code stops on one side of it, CUDA starts on the other.
//
// The transform is optional, and that choice picks one of two orderings on the
// stage's single stream:
//
//   with a transform          without one
//   -----------------------   ------------------------
//   H2D  pinned -> scratch    acquire output slot
//   release pinned slot       H2D  pinned -> that slot
//   acquire output slot       release pinned slot
//   transform scratch->slot   mark_slot_written
//   mark_slot_written
//   record scratch_free
class UploadStage {
 public:
  struct Config {
    uint32_t scratch_slots = 4;
    std::chrono::milliseconds pop_timeout{100};
    int device_id = 0;
    bool use_graph = false;
  };

  // `in`, `out` and `transform` must all outlive the stage. `out` must already
  // be sized for what the transform produces; the constructor checks and throws
  // rather than letting a too-small slot corrupt at the first frame.
  UploadStage(HostIngressRing& in, DeviceRingBuffer& out, DeviceTransform& transform,
              const ImageDesc& input_desc, Config config)
      : UploadStage(in, out, &transform, input_desc, config) {}

  // Upload only. The output ring receives the staged bytes unchanged, so its
  // slots are sized against input_desc and no scratch is allocated.
  UploadStage(HostIngressRing& in, DeviceRingBuffer& out, const ImageDesc& input_desc,
              Config config)
      : UploadStage(in, out, nullptr, input_desc, config) {}

  // Config's defaults cannot be spelled as a default argument here: GCC does
  // not have a nested class's member initialisers ready while the enclosing
  // class is still being parsed. Delegating sidesteps it.
  UploadStage(HostIngressRing& in, DeviceRingBuffer& out, DeviceTransform& transform,
              const ImageDesc& input_desc)
      : UploadStage(in, out, &transform, input_desc, Config{}) {}

  UploadStage(HostIngressRing& in, DeviceRingBuffer& out, const ImageDesc& input_desc)
      : UploadStage(in, out, nullptr, input_desc, Config{}) {}

  ~UploadStage();

  UploadStage(const UploadStage&) = delete;
  UploadStage& operator=(const UploadStage&) = delete;

  const ImageDesc& input_desc() const { return input_desc_; }
  const ImageDesc& output_desc() const { return output_desc_; }

  // Null on the upload-only path.
  const DeviceTransform* transform() const { return transform_; }
  bool has_transform() const { return transform_ != nullptr; }

  // Zero when there is no transform -- nothing to decouple, nothing allocated.
  uint32_t scratch_slots() const { return static_cast<uint32_t>(scratch_.size()); }

  bool graph_enabled() const { return !graph_exec_.empty(); }

  // Frames that took the eager path because the two ring indices had drifted
  // out of the lockstep the captured graphs assume. Should stay at zero; a
  // climbing count means graphs are silently off and something desynced them.
  uint64_t graph_fallbacks() const { return graph_fallbacks_.load(std::memory_order_relaxed); }

  // Take one staged frame through the four steps above. False if none arrived
  // within pop_timeout, or the ring stopped.
  //
  // Public so the stage can also be driven inline, without the worker, by a
  // producer that is already host-pinned and has nothing to overlap. Do not mix
  // the two -- either call step() yourself or call start(), never both, since
  // the stream and scratch ring are single-consumer.
  bool step();

  void start();
  void stop();
  bool running() const { return running_.load(std::memory_order_relaxed); }

  uint64_t uploaded() const { return uploaded_.load(std::memory_order_relaxed); }
  uint64_t failed() const { return failed_.load(std::memory_order_relaxed); }

 private:
  // Both public forms funnel here; a null transform selects the upload-only
  // ordering. Private so the nullable pointer stays an implementation detail
  // rather than something callers have to reason about.
  UploadStage(HostIngressRing& in, DeviceRingBuffer& out, DeviceTransform* transform,
              const ImageDesc& input_desc, Config config);

  void run();
  void release() noexcept;

  // Capture one graph per output slot, each baking that slot's pointer triple.
  void capture_graphs();
  // The shared tail of both paths: enqueue H2D + transform eagerly.
  void enqueue_eager(const HostIngressRing::Staged& staged, uint32_t scratch, void* dst);

  HostIngressRing* in_;
  DeviceRingBuffer* out_;
  DeviceTransform* transform_;
  ImageDesc input_desc_;
  ImageDesc output_desc_;
  Config config_;

  CudaStream stream_;
  std::vector<void*> scratch_;
  std::vector<cudaEvent_t> scratch_free_;

  // One executable graph per output slot; empty when graphs are off. Indexed by
  // the acquired output slot rather than a frame counter, so the index is
  // derived from state that cannot drift rather than tracked alongside it. The
  // scratch index is derived from that same slot, for the same reason.
  std::vector<cudaGraphExec_t> graph_exec_;

  std::thread worker_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> uploaded_{0};
  std::atomic<uint64_t> failed_{0};
  std::atomic<uint64_t> graph_fallbacks_{0};
};

}  // namespace perception
