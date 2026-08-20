#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "detection.hpp"
#include "transforms/yolo_preprocess.hpp"
#include "trt_engine.hpp"
#include "types.hpp"

namespace perception {

// Texture-read preprocess -> TensorRT inference, plus three ways to get the
// result out: decode() (host, blocking), draw_into() (GPU-only, rasterizes
// straight into a display surface), and the raw output_device() accessor
// (GPU-only, for a caller with its own kernel). Deliberately owns no ring,
// no consumer_id, and no thread: whoever calls enqueue() supplies the
// texture from whatever ReadLease it already holds, and everything after
// that is tied to exactly that call. That is what makes pairing an image
// with the boxes computed from it trivial for a caller -- present the same
// lease you just ran inference from, rather than reconciling two
// independently-paced consumers after the fact. See
// app/include/headless_yolo_consumer.hpp and
// app/include/yolo_viewer_consumer.hpp for the two things that add a ring
// consumer_id and a thread around this.
//
// Expects an end-to-end ("NMS-free") export: one input tensor shaped
// (1, 3, model_h, model_w) and one output shaped (1, N, 6) with rows
// [x1, y1, x2, y2, score, class_id] already in model input-pixel space --
// A model that still needs NMS applied externally is out of scope here.
class YoloEngine {
 public:
  struct Config {
    std::string engine_path;
    uint32_t model_width = 640;
    uint32_t model_height = 640;
    float conf_threshold = 0.25f;
    int device_id = 0;
  };

  YoloEngine(const ImageDesc& source_desc, Config config);
  ~YoloEngine();

  YoloEngine(const YoloEngine&) = delete;
  YoloEngine& operator=(const YoloEngine&) = delete;

  // Preprocess + TensorRT inference, enqueued on `stream`.
  void enqueue(cudaTextureObject_t texture, cudaStream_t stream);

  // Host path: copies the output tensor back and decodes it into
  // source_desc-pixel-space detections.
  std::vector<Detection> decode(cudaStream_t stream);

  // GPU-only path: rasterizes an outline for every detection scoring at
  // least conf_threshold directly into `surface` -- an RGBA8 CUDA surface
  // sized this engine's source_desc, e.g. the GL interop array
  // GlViewer::present_gpu_boxes maps for its frame copy -- without ever
  // copying the output tensor to the host.
  void draw_into(cudaSurfaceObject_t surface, cudaStream_t stream);

  // enqueue() + decode() together, for callers that only ever want the host
  // path (HeadlessYoloConsumer's Host output mode).
  std::vector<Detection> infer(cudaTextureObject_t texture, cudaStream_t stream) {
    enqueue(texture, stream);
    return decode(stream);
  }

  // Raw model output tensor, still in device memory, in model input-pixel
  // space (rows are [x1, y1, x2, y2, score, class_id], undecoded and
  // unfiltered) -- for a caller that wants to consume detections itself on
  // the GPU without ever copying them to the host, the same way
  // draw_into() does (see HeadlessYoloConsumer's Device output mode).
  // Overwritten by the next enqueue() cycle
  const float* output_device() const { return static_cast<const float*>(output_device_); }

  std::size_t max_detections() const { return max_detections_; }
  float conf_threshold() const { return config_.conf_threshold; }
  LetterboxParams letterbox() const {
    return preprocess_.letterbox_for(source_desc_.width, source_desc_.height);
  }

  double last_infer_ms() const { return last_infer_ms_; }

 private:
  ImageDesc source_desc_;
  Config config_;

  YoloPreprocessTransform preprocess_;
  std::unique_ptr<TrtEngine> engine_;

  std::string input_name_;
  std::string output_name_;
  std::size_t max_detections_ = 0;

  void* input_device_ = nullptr;
  void* output_device_ = nullptr;
  std::vector<float> output_host_;

  double last_infer_ms_ = 0.0;
};

}  // namespace perception
