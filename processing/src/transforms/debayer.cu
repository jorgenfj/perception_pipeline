#include "transforms/debayer.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "cuda_util.hpp"

namespace perception {
namespace {

struct BayerPhase {
  int red_x;
  int red_y;
};

BayerPhase phase_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::Bayer8_RGGB:
      return {0, 0};
    case PixelFormat::Bayer8_GRBG:
      return {1, 0};
    case PixelFormat::Bayer8_GBRG:
      return {0, 1};
    case PixelFormat::Bayer8_BGGR:
      return {1, 1};
    default:
      throw std::runtime_error("DebayerTransform: input is not an 8-bit Bayer format");
  }
}

// Mirror about the edge pixel. Reflection is the only cheap border rule that
// preserves parity -- both -v and 2(n-1)-v are congruent to v mod 2 -- so the
// mosaic phase survives it. Clamping does not: it would fold a red site onto a
// green one and smear the wrong channel along the border.
//
// Only ever called with v in [-1, n], which is all bilinear reaches, so a
// single reflection is enough and n >= 2 keeps the result in range.
__device__ __forceinline__ int reflect(int v, int n) {
  if (v < 0) return -v;
  if (v >= n) return 2 * (n - 1) - v;
  return v;
}

__global__ void debayer_bilinear_rgba8_kernel(const uint8_t* __restrict__ src, int src_stride,
                                              uint8_t* __restrict__ dst, int dst_stride, int width,
                                              int height, int red_x, int red_y) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) return;

  // __ldg: the mosaic is read up to nine times over by neighbouring threads and
  // never written, so it belongs in the read-only cache.
  const auto at = [&](int sx, int sy) -> int {
    return __ldg(&src[reflect(sy, height) * src_stride + reflect(sx, width)]);
  };

  // 0 where this coordinate is in phase with red, 1 where it is not. XOR
  // rather than subtraction because red_{x,y} are already 0 or 1.
  const int px = (x ^ red_x) & 1;
  const int py = (y ^ red_y) & 1;

  const int centre = at(x, y);
  int r, g, b;

  if (px == 0 && py == 0) {
    // Red site. The four green neighbours are axial, the four blue diagonal.
    r = centre;
    g = (at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) + 2) / 4;
    b = (at(x - 1, y - 1) + at(x + 1, y - 1) + at(x - 1, y + 1) + at(x + 1, y + 1) + 2) / 4;
  } else if (px == 1 && py == 1) {
    // Blue site. Mirror image of the above.
    b = centre;
    g = (at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) + 2) / 4;
    r = (at(x - 1, y - 1) + at(x + 1, y - 1) + at(x - 1, y + 1) + at(x + 1, y + 1) + 2) / 4;
  } else {
    // Green site, and there are two kinds. py == 0 puts it on a row that also
    // carries red, so its red neighbours are the horizontal pair and its blue
    // the vertical; on a blue row (py == 1) the two swap.
    g = centre;
    const int horizontal = (at(x - 1, y) + at(x + 1, y) + 1) / 2;
    const int vertical = (at(x, y - 1) + at(x, y + 1) + 1) / 2;
    r = (py == 0) ? horizontal : vertical;
    b = (py == 0) ? vertical : horizontal;
  }

  // One 4-byte store, and consecutive threads cover consecutive x, so a warp
  // writes a contiguous 128-byte line.
  *reinterpret_cast<uchar4*>(dst + static_cast<std::ptrdiff_t>(y) * dst_stride + x * 4) =
      make_uchar4(static_cast<unsigned char>(r), static_cast<unsigned char>(g),
                  static_cast<unsigned char>(b), 255);
}

}  // namespace

void DebayerTransform::enqueue(const void* src, const ImageDesc& input, void* dst,
                               const ImageDesc& output, cudaStream_t stream) {
  const BayerPhase phase = phase_of(input.format);

  if (output.format != PixelFormat::RGBA8) {
    throw std::runtime_error("DebayerTransform: output format must be RGBA8");
  }
  if (input.width != output.width || input.height != output.height) {
    throw std::runtime_error("DebayerTransform: input and output dimensions differ");
  }
  if (input.width < 2 || input.height < 2) {
    throw std::runtime_error("DebayerTransform: image must be at least 2x2");
  }
  if (input.stride_bytes < input.width) {
    throw std::runtime_error("DebayerTransform: input stride is shorter than a row");
  }
  if (output.stride_bytes < output.width * 4 || output.stride_bytes % 4 != 0) {
    throw std::runtime_error("DebayerTransform: output stride must be a 4-byte multiple of at "
                             "least width * 4");
  }

  const dim3 block(32, 8);
  const dim3 grid((input.width + block.x - 1) / block.x, (input.height + block.y - 1) / block.y);

  debayer_bilinear_rgba8_kernel<<<grid, block, 0, stream>>>(
      static_cast<const uint8_t*>(src), static_cast<int>(input.stride_bytes),
      static_cast<uint8_t*>(dst), static_cast<int>(output.stride_bytes),
      static_cast<int>(input.width), static_cast<int>(input.height), phase.red_x, phase.red_y);

  cuda_error_check(cudaGetLastError(), "DebayerTransform: kernel launch");
}

}  // namespace perception
