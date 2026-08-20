#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "cuda_util.hpp"
#include "detection.hpp"
#include "detection_box.hpp"
#include "device_ring_buffer.hpp"
#include "transforms/yolo_preprocess.hpp"
#include "types.hpp"
#include "yolo_config.hpp"
#include "yolo_engine.hpp"

namespace perception {

class HeadlessYoloConsumer {
 public:
  enum class OutputLocation {
    Host,
    Device,
  };

  // A single, reused buffer, not a ring: the next inference cycle
  // overwrites `output` as soon as it starts, ordered only against
  // `ready_event` on whatever stream last wrote it. A caller that is not
  // comfortably faster than this consumer's own frame rate can lose a race
  // with the next write
  struct DeviceOutput {
    const float* output = nullptr;
    std::size_t max_detections = 0;
    float conf_threshold = 0.0f;
    LetterboxParams letterbox{};
    // Wait on this (cudaStreamWaitEvent) before reading `output` from any
    // stream other than the one this consumer's worker thread used.
    cudaEvent_t ready_event = nullptr;
    uint32_t slot = 0;
    uint64_t slot_seq = 0;
  };

  HeadlessYoloConsumer(DeviceRingBuffer& ring, const ImageDesc& source_desc, const YoloConfig& config,
                       OutputLocation output_location, uint32_t consumer_id, int device_id);
  ~HeadlessYoloConsumer();

  HeadlessYoloConsumer(const HeadlessYoloConsumer&) = delete;
  HeadlessYoloConsumer& operator=(const HeadlessYoloConsumer&) = delete;

  void start();
  void stop();

  // Latest detections in host memory
  std::vector<DetectionBox> latest_boxes() const;

  // Latest detections in device memory
  DeviceOutput latest_device() const;

 private:
  bool step(CudaStream& stream);
  void run();

  DeviceRingBuffer* ring_;
  uint32_t consumer_id_;
  int device_id_;
  OutputLocation output_location_;
  YoloEngine engine_;
  CudaEvent ready_event_;

  std::thread worker_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> processed_{0};

  mutable std::mutex boxes_mutex_;
  std::vector<DetectionBox> boxes_;

  mutable std::mutex device_mutex_;
  DeviceOutput device_output_;

  uint32_t last_slot_ = 0;
  uint64_t last_seq_ = 0;
  bool have_last_ = false;
};

}  // namespace perception
