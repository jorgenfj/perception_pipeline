#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace perception {

// Rectify, resize and normalize in one pass, straight into an ESS input
// binding.

inline constexpr uint32_t kEssFullWidth = 960;
inline constexpr uint32_t kEssFullHeight = 576;

struct EssNormalization {
  float mean[3] = {0.485f, 0.456f, 0.406f};
  float stddev[3] = {0.229f, 0.224f, 0.225f};

  // What a pixel whose ray leaves the source image -- or looks behind the
  // camera -- gets, in the same [0, 1] the texture returns. The default is the
  // channel mean, which normalizes to zero.
  float no_source_value = 0.449f;
};

class EssPreprocessTransform {
 public:
  EssPreprocessTransform(const std::vector<float>& map, uint32_t source_width,
                         uint32_t source_height, uint32_t width, uint32_t height,
                         EssNormalization normalization = {});
  ~EssPreprocessTransform();

  EssPreprocessTransform(const EssPreprocessTransform&) = delete;
  EssPreprocessTransform& operator=(const EssPreprocessTransform&) = delete;

  std::size_t output_floats() const { return 3ull * width_ * height_; }
  std::size_t output_bytes() const { return output_floats() * sizeof(float); }

  // `src_texture` must be an RGBA8 texture read as cudaReadModeNormalizedFloat
  // with cudaFilterModeLinear and normalized coordinates 
  void enqueue(cudaTextureObject_t src_texture, float* dst, cudaStream_t stream) const;

  const char* name() const { return "EssPreprocess"; }

 private:
  float2* map_ = nullptr;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  EssNormalization normalization_;
};

}  // namespace perception
