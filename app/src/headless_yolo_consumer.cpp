#include "headless_yolo_consumer.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace perception {
namespace {

constexpr uint64_t kReportEvery = 30;
constexpr std::chrono::microseconds kPollInterval{2000};

YoloEngine::Config to_engine_config(const YoloConfig& config, int device_id) {
  YoloEngine::Config engine_config;
  engine_config.engine_path = config.engine_path;
  engine_config.model_width = config.model_width;
  engine_config.model_height = config.model_height;
  engine_config.conf_threshold = config.conf_threshold;
  engine_config.device_id = device_id;
  return engine_config;
}

}  // namespace

HeadlessYoloConsumer::HeadlessYoloConsumer(DeviceRingBuffer& ring, const ImageDesc& source_desc,
                                          const YoloConfig& config, OutputLocation output_location,
                                          uint32_t consumer_id, int device_id)
    : ring_(&ring),
      consumer_id_(consumer_id),
      device_id_(device_id),
      output_location_(output_location),
      engine_(source_desc, to_engine_config(config, device_id)) {
  if (consumer_id_ >= ring_->max_consumers()) {
    throw std::runtime_error(
        "HeadlessYoloConsumer: consumer id is outside the ring's max_consumers");
  }
}

HeadlessYoloConsumer::~HeadlessYoloConsumer() { stop(); }

void HeadlessYoloConsumer::start() {
  if (running_.exchange(true)) return;
  worker_ = std::thread([this] { run(); });
}

void HeadlessYoloConsumer::stop() {
  if (!running_.exchange(false)) return;
  if (worker_.joinable()) worker_.join();
}

std::vector<DetectionBox> HeadlessYoloConsumer::latest_boxes() const {
  std::lock_guard<std::mutex> lock(boxes_mutex_);
  return boxes_;
}

HeadlessYoloConsumer::DeviceOutput HeadlessYoloConsumer::latest_device() const {
  std::lock_guard<std::mutex> lock(device_mutex_);
  return device_output_;
}

bool HeadlessYoloConsumer::step(CudaStream& stream) {
  FramePeek peek;
  if (!ring_->view_latest_inplace(peek)) return false;
  if (have_last_ && peek.slot == last_slot_ && peek.slot_seq == last_seq_) return false;

  ReadLease lease = ring_->lease_latest(consumer_id_, stream);
  if (!lease.valid()) return false;

  cuda_error_check(cudaStreamWaitEvent(stream, lease.data_ready_event(), 0),
                   "HeadlessYoloConsumer: cudaStreamWaitEvent");

  engine_.enqueue(lease.texture(), stream);

  const uint32_t slot = lease.slot();
  const uint64_t seq = lease.seq();
  std::size_t found = 0;

  if (output_location_ == OutputLocation::Host) {
    const std::vector<Detection> detections = engine_.decode(stream);
    found = detections.size();

    std::vector<DetectionBox> converted;
    converted.reserve(detections.size());
    for (const Detection& det : detections) {
      converted.push_back(DetectionBox{det.x1, det.y1, det.x2, det.y2, det.score, det.class_id});
    }
    std::lock_guard<std::mutex> lock(boxes_mutex_);
    boxes_ = std::move(converted);
  } else {
    cuda_error_check(cudaEventRecord(ready_event_, stream), "HeadlessYoloConsumer: cudaEventRecord");

    std::lock_guard<std::mutex> lock(device_mutex_);
    device_output_.output = engine_.output_device();
    device_output_.max_detections = engine_.max_detections();
    device_output_.conf_threshold = engine_.conf_threshold();
    device_output_.letterbox = engine_.letterbox();
    device_output_.ready_event = ready_event_.get();
    device_output_.slot = slot;
    device_output_.slot_seq = seq;
  }

  last_slot_ = slot;
  last_seq_ = seq;
  have_last_ = true;
  lease.drop_hold();

  const uint64_t done = processed_.fetch_add(1, std::memory_order_relaxed) + 1;
  if (done % kReportEvery == 0) {
    if (output_location_ == OutputLocation::Host) {
      std::printf("yolo: processed=%lu infer=%.1fms detections=%zu\n",
                 static_cast<unsigned long>(done), engine_.last_infer_ms(), found);
    } else {
      std::printf("yolo: processed=%lu (device output)\n", static_cast<unsigned long>(done));
    }
  }
  return true;
}

void HeadlessYoloConsumer::run() {
  if (cudaSetDevice(device_id_) != cudaSuccess) {
    std::fprintf(stderr, "yolo: cudaSetDevice failed on the headless worker thread\n");
    running_.store(false, std::memory_order_relaxed);
    return;
  }

  CudaStream stream;
  while (running_.load(std::memory_order_relaxed)) {
    try {
      if (!step(stream)) std::this_thread::sleep_for(kPollInterval);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "yolo: %s\n", e.what());
    }
  }
  cudaStreamSynchronize(stream);
}

}  // namespace perception
