#include "transforms/rectification.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cuda_util.hpp"

namespace perception {
namespace {

// --- device: applying the map -------------------------------------------------

__global__ void rectify_lookup_rgba8_kernel(cudaTextureObject_t img,
                                            const float2* __restrict__ map,
                                            uint8_t* __restrict__ dst, int dst_stride, int width,
                                            int height) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) return;

  const std::size_t idx = static_cast<std::size_t>(y) * width + x;

  // The whole kernel, and the reason the map exists. Two units, no arithmetic:
  //
  // __ldcs -- streaming load. Every map entry is read exactly once per frame by
  // exactly one thread, so caching it evicts something that will be reused for
  // something that will not. Consecutive threads read consecutive entries, so a
  // warp's 32 loads are four 64-byte sectors.
  //
  // tex2D -- the TMU does the bilinear weighting and the border handling in
  // fixed function, off the ALU's critical path, and its own cache is what
  // absorbs the fact that neighbouring threads sample overlapping source
  // texels. This is the one access in the frame path that is deliberately not
  // coalesced: rectification scatters, and the TMU is the unit built for it.
  const float2 m = __ldcs(&map[idx]);              // LSU -- plain global load
  const float4 px = tex2D<float4>(img, m.x, m.y);  // TMU -- filtered fetch

  // cudaReadModeNormalizedFloat hands back [0, 1]; __saturatef costs nothing
  // and keeps a coordinate that sampled past the border from wrapping the cast.
  const auto to_u8 = [](float c) -> unsigned char {
    return static_cast<unsigned char>(__saturatef(c) * 255.0f + 0.5f);
  };

  *reinterpret_cast<uchar4*>(dst + static_cast<std::ptrdiff_t>(y) * dst_stride + x * 4) =
      make_uchar4(to_u8(px.x), to_u8(px.y), to_u8(px.z), 255);
}

}  // namespace

RectifyTransform::RectifyTransform(const std::vector<float2>& host_map, uint32_t width,
                                   uint32_t height, RectifyMapCoords coords)
    : width_(width), height_(height), coords_(coords) {
  if (width == 0 || height == 0) {
    throw std::runtime_error("RectifyTransform: zero-sized geometry");
  }
  if (host_map.size() != static_cast<std::size_t>(width) * height) {
    throw std::runtime_error("RectifyTransform: map has " + std::to_string(host_map.size()) +
                             " entries but " + std::to_string(width) + "x" +
                             std::to_string(height) + " needs " +
                             std::to_string(static_cast<std::size_t>(width) * height));
  }
  out_of_frame_ = count_out_of_frame(host_map, width, height, coords);

  const std::size_t bytes = host_map.size() * sizeof(float2);
  cuda_error_check(cudaMalloc(&map_, bytes), "RectifyTransform: cudaMalloc map");

  // Blocking, on the default stream, exactly once per eye at construction. The
  // frame path never touches this.
  const cudaError_t copy = cudaMemcpy(map_, host_map.data(), bytes, cudaMemcpyHostToDevice);
  if (copy != cudaSuccess) {
    cudaFree(map_);
    map_ = nullptr;
    cuda_error_check(copy, "RectifyTransform: cudaMemcpy map");
  }
}

RectifyTransform::~RectifyTransform() {
  if (map_) cudaFree(map_);
}

void RectifyTransform::enqueue(cudaTextureObject_t src_texture, void* dst, const ImageDesc& output,
                               cudaStream_t stream) const {
  if (src_texture == 0) {
    throw std::runtime_error(
        "RectifyTransform: null texture -- the source ring was not built with a "
        "DeviceRingTextureDesc");
  }
  if (output.format != PixelFormat::RGBA8) {
    throw std::runtime_error("RectifyTransform: output format must be RGBA8");
  }
  if (output.width != width_ || output.height != height_) {
    throw std::runtime_error("RectifyTransform: output is " + std::to_string(output.width) + "x" +
                             std::to_string(output.height) + " but the map was built for " +
                             std::to_string(width_) + "x" + std::to_string(height_));
  }
  if (output.stride_bytes < output.width * 4 || output.stride_bytes % 4 != 0) {
    throw std::runtime_error("RectifyTransform: output stride must be a 4-byte multiple of at "
                             "least width * 4");
  }

  const dim3 block(32, 8);
  const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);

  rectify_lookup_rgba8_kernel<<<grid, block, 0, stream>>>(
      src_texture, map_, static_cast<uint8_t*>(dst), static_cast<int>(output.stride_bytes),
      static_cast<int>(width_), static_cast<int>(height_));

  cuda_error_check(cudaGetLastError(), "RectifyTransform: kernel launch");
}

}  // namespace perception
