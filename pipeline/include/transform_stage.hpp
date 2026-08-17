#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#include "cuda_util.hpp"
#include "device_ring_buffer.hpp"
#include "device_transform.hpp"
#include "types.hpp"

namespace perception {

// A device-to-device hop: input ring -> transform -> output ring. The converter
// stage that rectify, resize and format conversion all instantiate.
//
// Per frame, on one non-blocking stream:
//
//   1. take the input ring's newest published view
//   2. cudaStreamWaitEvent on its readiness   (queue-side only, no host block)
//   3. acquire an output slot
//   4. transform.enqueue  input slot -> output slot
//   5. mark_slot_written, carrying the input frame's timestamp through
//
// Two things differ from UploadStage, both because the input is a device ring
// rather than a pinned queue:
//
//   * No scratch. The input slot is already device memory in the right place,
//     so the transform reads it directly and the stage allocates nothing but a
//     stream.
//   * No backpressure. The input is latest-wins, so there is nothing to pop and
//     nothing to release; a frame that arrives while this stage is busy simply
//     replaces the one before it. step() returns false when nothing new has
//     been published, and the worker polls.
class TransformStage {
 public:
  struct Config {
    // How long the worker sleeps when the input ring has published nothing new.
    std::chrono::microseconds poll_interval{200};
    // Which of the input ring's read-completion event slots this stage owns.
    // Must be unique among that ring's consumers and < its max_consumers().
    uint32_t consumer_id = 0;
    int device_id = 0;
  };

  // `in`, `out` and `transform` must all outlive the stage, and `out` must be
  // sized for what the transform produces -- checked here rather than left to
  // corrupt the first frame.
  TransformStage(DeviceRingBuffer& in, DeviceRingBuffer& out, DeviceTransform& transform,
                 const ImageDesc& input_desc, Config config);

  // Config's defaults cannot be spelled as a default argument here: GCC does
  // not have a nested class's member initialisers ready while the enclosing
  // class is still being parsed. Delegating sidesteps it.
  TransformStage(DeviceRingBuffer& in, DeviceRingBuffer& out, DeviceTransform& transform,
                 const ImageDesc& input_desc)
      : TransformStage(in, out, transform, input_desc, Config{}) {}

  ~TransformStage();

  TransformStage(const TransformStage&) = delete;
  TransformStage& operator=(const TransformStage&) = delete;

  const ImageDesc& input_desc() const { return input_desc_; }
  const ImageDesc& output_desc() const { return output_desc_; }
  const DeviceTransform& transform() const { return *transform_; }

  // Transform one newly published frame. False when the input ring has nothing
  // newer than the last frame this stage handled -- the normal idle result, not
  // an error.
  //
  // Public so the stage can be driven inline as well as by the worker. Do not
  // mix the two: the stream is single-consumer.
  bool step();

  void start();
  void stop();
  bool running() const { return running_.load(std::memory_order_relaxed); }

  uint64_t transformed() const { return transformed_.load(std::memory_order_relaxed); }
  uint64_t failed() const { return failed_.load(std::memory_order_relaxed); }

  // Times this stage could not get a lease on the newest frame because the
  // producer kept winning the race for it. Not corruption -- the frame is simply
  // skipped, which latest-wins allows. A climbing count means the ring is too
  // shallow or the producer too fast relative to this stage.
  uint64_t missed() const { return missed_.load(std::memory_order_relaxed); }

 private:
  void run();

  DeviceRingBuffer* in_;
  DeviceRingBuffer* out_;
  DeviceTransform* transform_;
  ImageDesc input_desc_;
  ImageDesc output_desc_;
  Config config_;

  CudaStream stream_;

  // What was last handled, so a re-published latest is not processed twice.
  // (slot, seq) rather than the timestamp, which a source is free to repeat.
  uint32_t last_slot_ = 0;
  uint64_t last_seq_ = 0;
  bool have_last_ = false;

  std::thread worker_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> transformed_{0};
  std::atomic<uint64_t> failed_{0};
  std::atomic<uint64_t> missed_{0};
};

}  // namespace perception
