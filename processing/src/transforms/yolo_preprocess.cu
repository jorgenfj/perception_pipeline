#include "transforms/yolo_preprocess.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "cuda_util.hpp"

namespace perception {
namespace {

__global__ void yolo_preprocess_kernel(cudaTextureObject_t tex, uint32_t model_w, uint32_t model_h,
                                       float pad_x, float pad_y, float new_w, float new_h,
                                       float pad_value, float* __restrict__ dst) {
  const int ox = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int oy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (ox >= static_cast<int>(model_w) || oy >= static_cast<int>(model_h)) return;

  const float fx = static_cast<float>(ox) - pad_x;
  const float fy = static_cast<float>(oy) - pad_y;

  float r = pad_value, g = pad_value, b = pad_value;
  if (fx >= 0.0f && fx < new_w && fy >= 0.0f && fy < new_h) {
    const float u = (fx + 0.5f) / new_w;
    const float v = (fy + 0.5f) / new_h;
    const float4 texel = tex2D<float4>(tex, u, v);
    r = texel.x;
    g = texel.y;
    b = texel.z;
  }

  const std::size_t plane = static_cast<std::size_t>(model_w) * model_h;
  const std::size_t idx = static_cast<std::size_t>(oy) * model_w + static_cast<std::size_t>(ox);
  dst[idx] = r;
  dst[plane + idx] = g;
  dst[2 * plane + idx] = b;
}

}  // namespace

LetterboxParams compute_letterbox(uint32_t src_w, uint32_t src_h, uint32_t model_w, uint32_t model_h) {
  if (src_w == 0 || src_h == 0 || model_w == 0 || model_h == 0) {
    throw std::runtime_error("compute_letterbox: zero-sized geometry");
  }
  const float scale = std::min(static_cast<float>(model_w) / static_cast<float>(src_w),
                               static_cast<float>(model_h) / static_cast<float>(src_h));
  const float new_w = static_cast<float>(src_w) * scale;
  const float new_h = static_cast<float>(src_h) * scale;
  LetterboxParams params;
  params.scale = scale;
  params.pad_x = (static_cast<float>(model_w) - new_w) * 0.5f;
  params.pad_y = (static_cast<float>(model_h) - new_h) * 0.5f;
  return params;
}

YoloPreprocessTransform::YoloPreprocessTransform(uint32_t model_width, uint32_t model_height,
                                                 float pad_value)
    : model_width_(model_width), model_height_(model_height), pad_value_(pad_value) {
  if (model_width_ == 0 || model_height_ == 0) {
    throw std::runtime_error("YoloPreprocessTransform: zero-sized model input");
  }
}

void YoloPreprocessTransform::enqueue(cudaTextureObject_t src_texture, uint32_t src_width,
                                      uint32_t src_height, float* dst, cudaStream_t stream) const {
  if (src_texture == 0) {
    throw std::runtime_error(
        "YoloPreprocessTransform: null texture -- the source ring was not built with a "
        "DeviceRingTextureDesc");
  }
  if (src_width == 0 || src_height == 0) {
    throw std::runtime_error("YoloPreprocessTransform: zero-sized source image");
  }

  const LetterboxParams lb = compute_letterbox(src_width, src_height, model_width_, model_height_);
  const float new_w = static_cast<float>(src_width) * lb.scale;
  const float new_h = static_cast<float>(src_height) * lb.scale;

  const dim3 block(32, 8);
  const dim3 grid((model_width_ + block.x - 1) / block.x, (model_height_ + block.y - 1) / block.y);

  yolo_preprocess_kernel<<<grid, block, 0, stream>>>(src_texture, model_width_, model_height_,
                                                     lb.pad_x, lb.pad_y, new_w, new_h, pad_value_,
                                                     dst);

  cuda_error_check(cudaGetLastError(), "YoloPreprocessTransform: kernel launch");
}

}  // namespace perception
