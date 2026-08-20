#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace perception {

struct LetterboxParams {
  float scale = 1.0f;
  float pad_x = 0.0f;
  float pad_y = 0.0f;
};

LetterboxParams compute_letterbox(uint32_t src_w, uint32_t src_h, uint32_t model_w, uint32_t model_h);

// Resizes and does the normalization to float in one pass using texture sampling.
// Input is RGBA8 HWC (cudaTextureObject_t) and output is CHW float array (float*)
class YoloPreprocessTransform {
 public:
  explicit YoloPreprocessTransform(uint32_t model_width, uint32_t model_height,
                                   float pad_value = 114.0f / 255.0f);

  uint32_t model_width() const { return model_width_; }
  uint32_t model_height() const { return model_height_; }

  std::size_t output_floats() const { return 3ull * model_width_ * model_height_; }
  std::size_t output_bytes() const { return output_floats() * sizeof(float); }

  LetterboxParams letterbox_for(uint32_t src_width, uint32_t src_height) const {
    return compute_letterbox(src_width, src_height, model_width_, model_height_);
  }

  // `src_texture` must be RGBA8 TextureObject.
  void enqueue(cudaTextureObject_t src_texture, uint32_t src_width, uint32_t src_height, float* dst,
               cudaStream_t stream) const;

 private:
  uint32_t model_width_;
  uint32_t model_height_;
  float pad_value_;
};

}  // namespace perception
