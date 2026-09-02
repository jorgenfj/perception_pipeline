#include "ess_engine.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "colorize_disparity.hpp"
#include "cuda_util.hpp"
#include "trt_plugins.hpp"

namespace perception {
namespace {

bool name_has(const std::string& name, const char* needle) {
  std::string lower = name;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return lower.find(needle) != std::string::npos;
}

std::string join_names(const std::vector<const TrtEngine::TensorInfo*>& tensors) {
  std::string joined;
  for (const TrtEngine::TensorInfo* info : tensors) {
    if (!joined.empty()) joined += ", ";
    joined += "'" + info->name + "'";
  }
  return joined;
}

const TrtEngine::TensorInfo* find_named(const std::vector<const TrtEngine::TensorInfo*>& tensors,
                                        const std::string& name) {
  for (const TrtEngine::TensorInfo* info : tensors) {
    if (info->name == name) return info;
  }
  return nullptr;
}

}  // namespace

EssEngine::EssEngine(const ImageDesc& source_desc, const utils::StereoCalibration& calibration,
                     Config config)
    : source_desc_(source_desc), config_(std::move(config)) {
  cuda_error_check(cudaSetDevice(config_.device_id), "EssEngine: cudaSetDevice");

  // The map's entries are source pixels of the calibrated grid, so a frame of a
  // different size samples the wrong place -- silently, and plausibly.
  if (source_desc_.width != calibration.size.width ||
      source_desc_.height != calibration.size.height) {
    throw std::runtime_error(
        "EssEngine: frames are " + std::to_string(source_desc_.width) + "x" +
        std::to_string(source_desc_.height) + " but the calibration was solved at " +
        std::to_string(calibration.size.width) + "x" + std::to_string(calibration.size.height));
  }

  for (int camera = 0; camera < 2; ++camera) {
    EssRectifier rectifier = make_ess_rectifier(calibration, camera, config_.normalization);
    preprocess_[camera] = std::move(rectifier.transform);
    if (camera == 0) {
      // Both cameras resize the same StereoRectification, so one copy is the pair's.
      rectification_ = rectifier.rectification;
      source_rectified_ = rectifier.source_rectified;
    }
  }
  sample_x_ = kEssFullWidth / 2;
  sample_y_ = kEssFullHeight / 2;

  // Before the TrtEngine: the ESS graph's fused ops come from a plugin whose
  // creators register from static initializers, so the library has to be
  // resident by the time the plan is deserialized.
  load_trt_plugins(config_.plugin_path);

  engine_ = std::make_unique<TrtEngine>(config_.engine_path, config_.device_id);

  std::vector<const TrtEngine::TensorInfo*> inputs;
  std::vector<const TrtEngine::TensorInfo*> outputs;
  for (const TrtEngine::TensorInfo& info : engine_->tensors()) {
    (info.is_input ? inputs : outputs).push_back(&info);
  }
  if (inputs.size() != 2) {
    throw std::runtime_error("EssEngine: expected two input tensors (left, right), found " +
                             std::to_string(inputs.size()) + ": " + join_names(inputs));
  }
  if (outputs.empty() || outputs.size() > 2) {
    throw std::runtime_error(
        "EssEngine: expected a disparity output and optionally a confidence one, found " +
        std::to_string(outputs.size()) + ": " + join_names(outputs));
  }
  if (config_.input_left_name.empty() != config_.input_right_name.empty()) {
    throw std::runtime_error("EssEngine: set both input_left_name and input_right_name, or neither");
  }
  if (!config_.input_left_name.empty()) {
    if (!find_named(inputs, config_.input_left_name) ||
        !find_named(inputs, config_.input_right_name)) {
      throw std::runtime_error("EssEngine: no input tensor named '" + config_.input_left_name +
                               "' and '" + config_.input_right_name + "' -- the engine has " +
                               join_names(inputs));
    }
    input_left_name_ = config_.input_left_name;
    input_right_name_ = config_.input_right_name;
  } else {
    for (const TrtEngine::TensorInfo* info : inputs) {
      if (name_has(info->name, "left")) input_left_name_ = info->name;
      if (name_has(info->name, "right")) input_right_name_ = info->name;
    }
    if (input_left_name_.empty() || input_right_name_.empty() ||
        input_left_name_ == input_right_name_) {
      throw std::runtime_error(
          "EssEngine: cannot tell the cameras apart from the input names " + join_names(inputs) +
          " -- set input_left_name / input_right_name");
    }
  }

  const std::size_t input_floats = preprocess_[0]->output_floats();
  for (const TrtEngine::TensorInfo* info : inputs) {
    if (info->element_count() != input_floats) {
      throw std::runtime_error("EssEngine: input '" + info->name + "' has " +
                               std::to_string(info->element_count()) + " elements, expected " +
                               std::to_string(input_floats) + " for the full ESS model's 1x3x" +
                               std::to_string(kEssFullHeight) + "x" +
                               std::to_string(kEssFullWidth) + " input -- the light model is not "
                               "supported");
    }
  }

  const std::size_t pixel_count = pixels();
  for (const TrtEngine::TensorInfo* info : outputs) {
    if (info->element_count() != pixel_count) {
      throw std::runtime_error("EssEngine: output '" + info->name + "' has " +
                               std::to_string(info->element_count()) + " elements, expected " +
                               std::to_string(pixel_count) + " (one value per network pixel)");
    }
    std::string& slot = name_has(info->name, "conf") ? confidence_name_ : disparity_name_;
    if (!slot.empty()) {
      throw std::runtime_error("EssEngine: cannot tell the disparity output from the confidence "
                               "one among " + join_names(outputs) +
                               " -- exactly one of them has to name itself a confidence map");
    }
    slot = info->name;
  }
  if (disparity_name_.empty()) {
    throw std::runtime_error("EssEngine: no disparity output among " + join_names(outputs));
  }

  for (int camera = 0; camera < 2; ++camera) {
    cuda_error_check(cudaMalloc(&input_device_[camera], preprocess_[camera]->output_bytes()),
                     "EssEngine: cudaMalloc input");
  }
  cuda_error_check(cudaMalloc(&disparity_device_, pixel_count * sizeof(float)),
                   "EssEngine: cudaMalloc disparity");
  if (!confidence_name_.empty()) {
    cuda_error_check(cudaMalloc(&confidence_device_, pixel_count * sizeof(float)),
                     "EssEngine: cudaMalloc confidence");
  }
  // Pinned: a D2H into pageable memory would synchronise the stream, which is
  // the one thing the sample path exists to avoid.
  cuda_error_check(cudaMallocHost(&sample_host_, kTimingSlots * 2 * sizeof(float)),
                   "EssEngine: cudaMallocHost sample");
  for (std::size_t i = 0; i < kTimingSlots * 2; ++i) sample_host_[i] = 0.0f;

  engine_->set_tensor_address(input_left_name_, input_device_[0]);
  engine_->set_tensor_address(input_right_name_, input_device_[1]);
  engine_->set_tensor_address(disparity_name_, disparity_device_);
  if (confidence_device_) engine_->set_tensor_address(confidence_name_, confidence_device_);

  const utils::CameraIntrinsics k = rectification_.rectified_intrinsics();
  std::printf("ess: %s, in=%s/%s out=%s%s%s, %ux%u cropped from %ux%u (scale %.3f), fx=%.1f "
             "baseline=%.4fm, conf>=%.2f\n",
             config_.engine_path.c_str(), input_left_name_.c_str(), input_right_name_.c_str(),
             disparity_name_.c_str(), confidence_device_ ? "+" : "",
             confidence_device_ ? confidence_name_.c_str() : "", width(), height(),
             source_rectified_.width, source_rectified_.height, disparity_scale(), k.fx,
             rectification_.baseline_m(), config_.conf_threshold);
  char far_depth[32] = "infinity";
  if (config_.display_min_disparity > 0.0f) {
    std::snprintf(far_depth, sizeof(far_depth), "%.2fm", depth_m(config_.display_min_disparity));
  }
  std::printf("ess: colormap %s over %.1f..%.1fpx, i.e. %.2fm (red) out to %s (blue)\n",
             config_.colormap == DisparityColormap::Jet ? "jet" : "turbo",
             config_.display_min_disparity, config_.display_max_disparity,
             depth_m(config_.display_max_disparity), far_depth);
}

EssEngine::~EssEngine() {
  for (void* input : input_device_) {
    if (input) cudaFree(input);
  }
  if (disparity_device_) cudaFree(disparity_device_);
  if (confidence_device_) cudaFree(confidence_device_);
  if (sample_host_) cudaFreeHost(sample_host_);
}

void EssEngine::set_sample_pixel(uint32_t x, uint32_t y) {
  if (x >= width() || y >= height()) {
    throw std::runtime_error("EssEngine: sample pixel (" + std::to_string(x) + ", " +
                             std::to_string(y) + ") is outside the network's " +
                             std::to_string(width()) + "x" + std::to_string(height()) + " grid");
  }
  sample_x_ = x;
  sample_y_ = y;
}

void EssEngine::preprocess(cudaTextureObject_t left_texture, cudaTextureObject_t right_texture,
                           cudaStream_t stream) {
  Timing& timing = timing_[enqueued_ % kTimingSlots];
  timing.armed = false;
  timing.drawn = false;

  cuda_error_check(cudaEventRecord(timing.pre_start, stream), "EssEngine: cudaEventRecord pre_start");
  preprocess_[0]->enqueue(left_texture, static_cast<float*>(input_device_[0]), stream);
  preprocess_[1]->enqueue(right_texture, static_cast<float*>(input_device_[1]), stream);
  preprocessed_ = true;
}

void EssEngine::infer(cudaStream_t stream) {
  if (!preprocessed_) {
    throw std::runtime_error(
        "EssEngine: infer() without a preceding preprocess() -- the input bindings still hold "
        "the previous cycle's frames");
  }
  preprocessed_ = false;

  Timing& timing = timing_[enqueued_ % kTimingSlots];
  cuda_error_check(cudaEventRecord(timing.infer_start, stream),
                   "EssEngine: cudaEventRecord infer_start");
  engine_->enqueue(stream);
  cuda_error_check(cudaEventRecord(timing.infer_end, stream),
                   "EssEngine: cudaEventRecord infer_end");

  float* slot = sample_host_ + (enqueued_ % kTimingSlots) * 2;
  const std::size_t offset = static_cast<std::size_t>(sample_y_) * width() + sample_x_;
  cuda_error_check(cudaMemcpyAsync(slot, static_cast<const float*>(disparity_device_) + offset,
                                   sizeof(float), cudaMemcpyDeviceToHost, stream),
                   "EssEngine: cudaMemcpyAsync disparity sample");
  if (confidence_device_) {
    cuda_error_check(cudaMemcpyAsync(slot + 1, static_cast<const float*>(confidence_device_) + offset,
                                     sizeof(float), cudaMemcpyDeviceToHost, stream),
                     "EssEngine: cudaMemcpyAsync confidence sample");
  } else {
    slot[1] = 0.0f;
  }
  cuda_error_check(cudaEventRecord(timing.sample_done, stream),
                   "EssEngine: cudaEventRecord sample_done");

  timing.sample_x = sample_x_;
  timing.sample_y = sample_y_;
  timing.armed = true;
  ++enqueued_;
}

bool EssEngine::last_timing(FrameTiming& out) const {
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

bool EssEngine::last_sample(Sample& out) const {
  for (uint64_t back = 1; back <= kTimingSlots && back <= enqueued_; ++back) {
    const uint64_t index = enqueued_ - back;
    const Timing& timing = timing_[index % kTimingSlots];
    if (!timing.armed) continue;

    if (cudaEventQuery(timing.sample_done) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }

    const float* slot = sample_host_ + (index % kTimingSlots) * 2;
    out.x = timing.sample_x;
    out.y = timing.sample_y;
    out.disparity_px = slot[0];
    out.confidence = slot[1];
    out.depth_m = depth_m(slot[0]);
    out.trusted = slot[0] > 0.0f &&
                  (confidence_device_ == nullptr || slot[1] >= config_.conf_threshold);
    out.index = index;
    return true;
  }
  return false;
}

void EssEngine::draw_into(cudaSurfaceObject_t surface, cudaStream_t stream) {
  colorize_disparity(static_cast<const float*>(disparity_device_),
                     static_cast<const float*>(confidence_device_), width(), height(),
                     config_.display_min_disparity, config_.display_max_disparity,
                     config_.conf_threshold, config_.colormap, surface, stream);

  // Closes the timing for the enqueue() this draw belongs to. Callers that
  // never draw leave total_ms ending at inference, which included_draw reports.
  if (enqueued_ > 0) {
    Timing& timing = timing_[(enqueued_ - 1) % kTimingSlots];
    if (timing.armed) {
      cuda_error_check(cudaEventRecord(timing.draw_end, stream),
                       "EssEngine: cudaEventRecord draw_end");
      timing.drawn = true;
    }
  }
}

const std::vector<float>& EssEngine::download(cudaStream_t stream) {
  // Here rather than in the constructor: 2 MB a map, and the GPU-only callers
  // -- the viewer, and the pair callback's sample -- never touch either.
  disparity_host_.resize(pixels());
  if (confidence_device_) confidence_host_.resize(pixels());

  const std::size_t bytes = pixels() * sizeof(float);
  cuda_error_check(cudaMemcpyAsync(disparity_host_.data(), disparity_device_, bytes,
                                   cudaMemcpyDeviceToHost, stream),
                   "EssEngine: cudaMemcpyAsync disparity");
  if (confidence_device_) {
    cuda_error_check(cudaMemcpyAsync(confidence_host_.data(), confidence_device_, bytes,
                                     cudaMemcpyDeviceToHost, stream),
                     "EssEngine: cudaMemcpyAsync confidence");
  }
  cuda_error_check(cudaStreamSynchronize(stream), "EssEngine: cudaStreamSynchronize");
  return disparity_host_;
}

}  // namespace perception
