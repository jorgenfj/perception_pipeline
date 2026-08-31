#include "transforms/ess_preprocess.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cuda_util.hpp"

namespace perception {
namespace {

__global__ void ess_preprocess_kernel(cudaTextureObject_t img, const float2* __restrict__ map,
                                      float* __restrict__ dst, int width, int height,
                                      float3 mean, float3 inv_stddev, float3 no_source) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) return;

  const std::size_t idx = static_cast<std::size_t>(y) * width + x;

  const float2 m = __ldcs(&map[idx]);              // LSU -- plain global load
  const float4 px = tex2D<float4>(img, m.x, m.y);  // TMU -- filtered fetch

  // An entry off the image still sampled something, because the texture unit
  // clamps to the border texel rather than failing -- but that something is
  // replicated edge, not image.
  const bool has_source = m.x >= 0.0f && m.x < 1.0f && m.y >= 0.0f && m.y < 1.0f;

  const std::size_t plane = static_cast<std::size_t>(width) * height;
  dst[idx] = has_source ? (px.x - mean.x) * inv_stddev.x : no_source.x;
  dst[plane + idx] = has_source ? (px.y - mean.y) * inv_stddev.y : no_source.y;
  dst[2 * plane + idx] = has_source ? (px.z - mean.z) * inv_stddev.z : no_source.z;
}

}  // namespace

EssPreprocessTransform::EssPreprocessTransform(const std::vector<float>& host_map,
                                               uint32_t source_width, uint32_t source_height,
                                               uint32_t width, uint32_t height,
                                               EssNormalization normalization)
    : width_(width), height_(height), normalization_(normalization) {
  if (width == 0 || height == 0 || source_width == 0 || source_height == 0) {
    throw std::runtime_error("EssPreprocessTransform: zero-sized geometry");
  }
  const std::size_t expected = 2 * static_cast<std::size_t>(width) * height;
  if (host_map.size() != expected) {
    throw std::runtime_error("EssPreprocessTransform: map has " +
                             std::to_string(host_map.size()) + " floats but " +
                             std::to_string(width) + "x" + std::to_string(height) + " needs " +
                             std::to_string(expected));
  }
  for (int c = 0; c < 3; ++c) {
    if (!(normalization.stddev[c] > 0.0f)) {
      throw std::runtime_error("EssPreprocessTransform: normalization stddev[" +
                               std::to_string(c) + "] must be positive, got " +
                               std::to_string(normalization.stddev[c]));
    }
  }

  // The map arrives in source pixels; the textures it will sample are in
  // normalized coordinates.
  std::vector<float> normalized(host_map.size());
  for (std::size_t i = 0; i + 1 < host_map.size(); i += 2) {
    normalized[i] = host_map[i] / static_cast<float>(source_width);
    normalized[i + 1] = host_map[i + 1] / static_cast<float>(source_height);
  }

  const std::size_t bytes = normalized.size() * sizeof(float);
  cuda_error_check(cudaMalloc(&map_, bytes), "EssPreprocessTransform: cudaMalloc map");

  const cudaError_t copy = cudaMemcpy(map_, normalized.data(), bytes, cudaMemcpyHostToDevice);
  if (copy != cudaSuccess) {
    cudaFree(map_);
    map_ = nullptr;
    cuda_error_check(copy, "EssPreprocessTransform: cudaMemcpy map");
  }
}

EssPreprocessTransform::~EssPreprocessTransform() {
  if (map_) cudaFree(map_);
}

void EssPreprocessTransform::enqueue(cudaTextureObject_t src_texture, float* dst,
                                     cudaStream_t stream) const {
  if (src_texture == 0) {
    throw std::runtime_error(
        "EssPreprocessTransform: null texture -- the source ring was not built with a "
        "DeviceRingTextureDesc");
  }
  if (dst == nullptr) {
    throw std::runtime_error("EssPreprocessTransform: null output tensor");
  }
  cudaTextureDesc tex_desc{};
  cuda_error_check(cudaGetTextureObjectTextureDesc(&tex_desc, src_texture),
                   "EssPreprocessTransform: cudaGetTextureObjectTextureDesc");
  if (tex_desc.normalizedCoords == 0) {
    throw std::runtime_error(
        "EssPreprocessTransform: the texture samples in pixel coordinates; this needs a ring "
        "built with DeviceRingTextureDesc::normalized_coords");
  }

  const dim3 block(32, 8);
  const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);

  const float3 mean =
      make_float3(normalization_.mean[0], normalization_.mean[1], normalization_.mean[2]);
  const float3 inv_stddev = make_float3(1.0f / normalization_.stddev[0],
                                        1.0f / normalization_.stddev[1],
                                        1.0f / normalization_.stddev[2]);

  const float3 no_source = make_float3((normalization_.no_source_value - mean.x) * inv_stddev.x,
                                       (normalization_.no_source_value - mean.y) * inv_stddev.y,
                                       (normalization_.no_source_value - mean.z) * inv_stddev.z);

  ess_preprocess_kernel<<<grid, block, 0, stream>>>(src_texture, map_, dst,
                                                    static_cast<int>(width_),
                                                    static_cast<int>(height_), mean, inv_stddev,
                                                    no_source);

  cuda_error_check(cudaGetLastError(), "EssPreprocessTransform: kernel launch");
}

}  // namespace perception
