#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <perception/geometry/stereo.hpp>

#include "colorize_disparity.hpp"
#include "cuda_util.hpp"
#include "ess_config.hpp"
#include "transforms/ess_preprocess.hpp"
#include "transforms/rectify_map.hpp"
#include "trt_engine.hpp"
#include "types.hpp"

namespace perception {

// Rectify + resize + normalize both eyes -> TensorRT -> disparity
//
// Expects the full ESS export: two inputs shaped (1, 3, 576, 960), a disparity
// output of 960*576 in network pixels, and optionally a confidence output the
// same size.
class EssEngine {
 public:
  struct Config {
    std::string engine_path;
    std::string plugin_path;
    EssNormalization normalization;
    float conf_threshold = 0.0f;
    int device_id = 0;

    // draw_into() only.
    DisparityColormap colormap = DisparityColormap::Turbo;
    float display_min_disparity = 0.0f;
    float display_max_disparity = 64.0f;

    std::string input_left_name;
    std::string input_right_name;
  };


  EssEngine(const ImageDesc& source_desc, const geometry::StereoCalibration& calibration,
            Config config);
  ~EssEngine();

  EssEngine(const EssEngine&) = delete;
  EssEngine& operator=(const EssEngine&) = delete;

  void preprocess(cudaTextureObject_t left_texture, cudaTextureObject_t right_texture,
                  cudaStream_t stream);

  void infer(cudaStream_t stream);

  // Host path, in network pixels. Blocks until the copy retires
  const std::vector<float>& download(cudaStream_t stream);

  // For the cycle download() last fetched; empty without a confidence output.
  const std::vector<float>& confidence_host() const { return confidence_host_; }

  // GPU-only path: paints the disparity into `surface`, an RGBA8 CUDA surface
  // sized width() x height()
  void draw_into(cudaSurfaceObject_t surface, cudaStream_t stream);

  // Device-side, row-major width() x height(), in network pixels. Overwritten
  // by the next infer() cycle; confidence is null without that output.
  const float* disparity_device() const { return static_cast<const float*>(disparity_device_); }
  const float* confidence_device() const { return static_cast<const float*>(confidence_device_); }

  uint32_t width() const { return rectification_.size.width; }
  uint32_t height() const { return rectification_.size.height; }
  std::size_t pixels() const { return static_cast<std::size_t>(width()) * height(); }

  // Restated on the network's grid
  const geometry::StereoRectification& rectification() const { return rectification_; }

  // A disparity in network pixels divided by this is one in calibrated pixels.
  double disparity_scale() const {
    return geometry::resize_scale(source_rectified_, rectification_.size,
                                  geometry::ResizeFit::Crop);
  }

  // m = fx * baseline / disparity
  double depth_m(float disparity_px) const {
    if (!(disparity_px > 0.0f)) return 0.0;
    return rectification_.rectified_fx() * rectification_.baseline_m() / disparity_px;
  }

  struct FrameTiming {
    double preprocess_ms = 0.0;  // both eyes' rectify + resize + normalize
    double inference_ms = 0.0;   // TensorRT execution, and nothing else
    double total_ms = 0.0;       // preprocess start through the end of drawing
    bool included_draw = false;  // false means total_ms stops at inference
    uint64_t index = 0;          // which infer() cycle this describes, for dedup
  };

  // Most recent cycle that has actually retired on the device, which is not the
  // most recent one enqueued. False until at least one cycle has completed.
  bool last_timing(FrameTiming& out) const;

  struct Sample {
    uint32_t x = 0;
    uint32_t y = 0;
    float disparity_px = 0.0f;
    float confidence = 0.0f;  // 0 when the engine has no confidence output
    double depth_m = 0.0;
    bool trusted = false;  // positive disparity, and confidence past the threshold
    uint64_t index = 0;
  };

  // One pixel, copied back into pinned memory with the cycle it belongs to and
  // read only once that copy has retired -- so this never syncs, and a caller
  // holding ring leases can still print a number.
  bool last_sample(Sample& out) const;

  // Defaults to the centre.
  void set_sample_pixel(uint32_t x, uint32_t y);

 private:
  ImageDesc source_desc_;
  Config config_;

  // One per eye; each carries its own map
  std::unique_ptr<EssPreprocessTransform> preprocess_[2];

  // Resized onto the network's grid, and the grid it was resized from.
  geometry::StereoRectification rectification_;
  geometry::ImageSize source_rectified_;

  std::unique_ptr<TrtEngine> engine_;

  std::string input_left_name_;
  std::string input_right_name_;
  std::string disparity_name_;
  std::string confidence_name_;  // empty when the engine has no confidence output

  void* input_device_[2] = {nullptr, nullptr};
  void* disparity_device_ = nullptr;
  void* confidence_device_ = nullptr;
  std::vector<float> disparity_host_;
  std::vector<float> confidence_host_;

  bool preprocessed_ = false;  // preprocess() ran, infer() has not consumed it

  uint32_t sample_x_ = 0;
  uint32_t sample_y_ = 0;
  // Pinned, two floats per slot: disparity then confidence.
  float* sample_host_ = nullptr;

  // Deep enough to outlast any sane host-ahead-of-GPU gap; the events are cheap
  // and only completed slots are ever read.
  static constexpr std::size_t kTimingSlots = 8;

  struct Timing {
    CudaEvent pre_start{cudaEventDefault};
    CudaEvent infer_start{cudaEventDefault};
    CudaEvent infer_end{cudaEventDefault};
    CudaEvent draw_end{cudaEventDefault};
    CudaEvent sample_done;  // queried, never measured -- timing stays off
    bool armed = false;  // infer() has recorded into this slot
    bool drawn = false;  // ... and draw_into() closed it
    uint32_t sample_x = 0;
    uint32_t sample_y = 0;
  };

  std::array<Timing, kTimingSlots> timing_;
  uint64_t enqueued_ = 0;
};

}  // namespace perception
