#include "yolo_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "cuda_util.hpp"
#include "draw_detections.hpp"

namespace perception {
namespace {

constexpr std::size_t kBoxStride = 6;  // x1, y1, x2, y2, score, class_id

}  // namespace

YoloEngine::YoloEngine(const ImageDesc& source_desc, Config config)
    : source_desc_(source_desc),
      config_(config),
      preprocess_(config.model_width, config.model_height) {
  cuda_error_check(cudaSetDevice(config_.device_id), "YoloEngine: cudaSetDevice");

  engine_ = std::make_unique<TrtEngine>(config_.engine_path, config_.device_id);

  for (const TrtEngine::TensorInfo& info : engine_->tensors()) {
    if (info.is_input) {
      if (!input_name_.empty()) {
        throw std::runtime_error("YoloEngine: engine has more than one input tensor");
      }
      input_name_ = info.name;
      if (info.element_count() != preprocess_.output_floats()) {
        throw std::runtime_error("YoloEngine: engine input '" + info.name + "' has " +
                                 std::to_string(info.element_count()) + " elements, expected " +
                                 std::to_string(preprocess_.output_floats()) + " for a 1x3x" +
                                 std::to_string(config_.model_height) + "x" +
                                 std::to_string(config_.model_width) + " input");
      }
    } else {
      if (!output_name_.empty()) {
        throw std::runtime_error("YoloEngine: engine has more than one output tensor");
      }
      output_name_ = info.name;
      if (info.element_count() % kBoxStride != 0) {
        throw std::runtime_error(
            "YoloEngine: engine output '" + info.name + "' has " +
            std::to_string(info.element_count()) +
            " elements, not a multiple of 6 -- expected an end-to-end export with rows "
            "[x1, y1, x2, y2, score, class_id]");
      }
      max_detections_ = info.element_count() / kBoxStride;
    }
  }
  if (input_name_.empty() || output_name_.empty()) {
    throw std::runtime_error("YoloEngine: engine needs exactly one input and one output tensor");
  }

  cuda_error_check(cudaMalloc(&input_device_, preprocess_.output_bytes()),
                   "YoloEngine: cudaMalloc input");
  cuda_error_check(cudaMalloc(&output_device_, max_detections_ * kBoxStride * sizeof(float)),
                   "YoloEngine: cudaMalloc output");
  output_host_.resize(max_detections_ * kBoxStride);

  engine_->set_tensor_address(input_name_, input_device_);
  engine_->set_tensor_address(output_name_, output_device_);

  std::printf("yolo: %s, input=%s output=%s, %ux%u, max_det=%zu, conf>=%.2f\n",
             config_.engine_path.c_str(), input_name_.c_str(), output_name_.c_str(),
             config_.model_width, config_.model_height, max_detections_, config_.conf_threshold);
}

YoloEngine::~YoloEngine() {
  if (input_device_) cudaFree(input_device_);
  if (output_device_) cudaFree(output_device_);
}

void YoloEngine::enqueue(cudaTextureObject_t texture, cudaStream_t stream) {
  Timing& timing = timing_[enqueued_ % kTimingSlots];
  timing.armed = false;
  timing.drawn = false;

  cuda_error_check(cudaEventRecord(timing.pre_start, stream),
                   "YoloEngine: cudaEventRecord pre_start");
  preprocess_.enqueue(texture, source_desc_.width, source_desc_.height,
                     static_cast<float*>(input_device_), stream);
  cuda_error_check(cudaEventRecord(timing.infer_start, stream),
                   "YoloEngine: cudaEventRecord infer_start");
  engine_->enqueue(stream);
  cuda_error_check(cudaEventRecord(timing.infer_end, stream),
                   "YoloEngine: cudaEventRecord infer_end");

  timing.armed = true;
  ++enqueued_;
}

bool YoloEngine::last_timing(FrameTiming& out) const {
  // Newest first. Anything older has necessarily retired too, so the first
  // complete slot found is also the freshest usable one.
  for (uint64_t back = 1; back <= kTimingSlots && back <= enqueued_; ++back) {
    const Timing& timing = timing_[(enqueued_ - back) % kTimingSlots];
    if (!timing.armed) continue;

    // A slot enqueued but not yet drawn measures to inference; on the newest
    // slot that can mean the draw is merely still to come, but that slot is
    // almost never complete anyway, so it gets skipped below.
    const cudaEvent_t last = timing.drawn ? timing.draw_end.get() : timing.infer_end.get();
    if (cudaEventQuery(last) != cudaSuccess) {
      cudaGetLastError();  // swallow the cudaErrorNotReady we just provoked
      continue;
    }

    float pre_ms = 0.0f;
    float infer_ms = 0.0f;
    float total_ms = 0.0f;
    if (cudaEventElapsedTime(&pre_ms, timing.pre_start, timing.infer_start) != cudaSuccess ||
        cudaEventElapsedTime(&infer_ms, timing.infer_start, timing.infer_end) != cudaSuccess ||
        cudaEventElapsedTime(&total_ms, timing.pre_start, last) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }

    out.preprocess_ms = static_cast<double>(pre_ms);
    out.inference_ms = static_cast<double>(infer_ms);
    out.total_ms = static_cast<double>(total_ms);
    out.included_draw = timing.drawn;
    out.index = enqueued_ - back;
    return true;
  }
  return false;
}

std::vector<Detection> YoloEngine::decode(cudaStream_t stream) {
  const auto started = std::chrono::steady_clock::now();

  // If data is needed on host should this copy be baked into the enqueue above
  // and then the host syncs/waits against that there instead?
  cuda_error_check(cudaMemcpyAsync(output_host_.data(), output_device_,
                                   output_host_.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   stream),
                   "YoloEngine: cudaMemcpyAsync output");

  // Decoding on the CPU needs the copy finished.
  cuda_error_check(cudaStreamSynchronize(stream), "YoloEngine: cudaStreamSynchronize");

  const LetterboxParams lb = preprocess_.letterbox_for(source_desc_.width, source_desc_.height);
  const float max_x = static_cast<float>(source_desc_.width - 1);
  const float max_y = static_cast<float>(source_desc_.height - 1);

  std::vector<Detection> detections;
  detections.reserve(16);
  for (std::size_t i = 0; i < max_detections_; ++i) {
    const float* row = &output_host_[i * kBoxStride];
    const float score = row[4];
    if (score < config_.conf_threshold) continue;

    Detection det;
    det.x1 = std::clamp((row[0] - lb.pad_x) / lb.scale, 0.0f, max_x);
    det.y1 = std::clamp((row[1] - lb.pad_y) / lb.scale, 0.0f, max_y);
    det.x2 = std::clamp((row[2] - lb.pad_x) / lb.scale, 0.0f, max_x);
    det.y2 = std::clamp((row[3] - lb.pad_y) / lb.scale, 0.0f, max_y);
    det.score = score;
    det.class_id = static_cast<int>(row[5]);
    detections.push_back(det);
  }

  last_infer_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return detections;
}

void YoloEngine::draw_into(cudaSurfaceObject_t surface, cudaStream_t stream) {
  const LetterboxParams lb = preprocess_.letterbox_for(source_desc_.width, source_desc_.height);
  draw_detections(static_cast<const float*>(output_device_), static_cast<uint32_t>(max_detections_),
                 config_.conf_threshold, lb.scale, lb.pad_x, lb.pad_y, source_desc_.width,
                 source_desc_.height, surface, stream);

  // Closes the timing for the enqueue() this draw belongs to. Callers that
  // never draw -- decode(), the device-output path -- simply leave total_ms
  // ending at inference, which included_draw reports.
  if (enqueued_ > 0) {
    Timing& timing = timing_[(enqueued_ - 1) % kTimingSlots];
    if (timing.armed) {
      cuda_error_check(cudaEventRecord(timing.draw_end, stream),
                       "YoloEngine: cudaEventRecord draw_end");
      timing.drawn = true;
    }
  }
}

}  // namespace perception
