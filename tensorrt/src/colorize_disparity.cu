#include "colorize_disparity.hpp"

#include "cuda_util.hpp"

namespace perception {
namespace {

__device__ float saturate_f(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }

// Google's published polynomial fit to Turbo -- two dot products per channel,
// which is cheaper here than a 256-entry lookup and its texture fetch.
__device__ float3 turbo(float x) {
  const float4 kRed4 = make_float4(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f);
  const float4 kGreen4 = make_float4(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f);
  const float4 kBlue4 = make_float4(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f);
  const float2 kRed2 = make_float2(-152.94239396f, 59.28637943f);
  const float2 kGreen2 = make_float2(4.27729857f, 2.82956604f);
  const float2 kBlue2 = make_float2(-89.90310912f, 27.34824973f);

  const float x2 = x * x;
  const float x3 = x2 * x;
  const float4 v4 = make_float4(1.0f, x, x2, x3);
  const float2 v2 = make_float2(x2 * x2, x2 * x3);  // x^4, x^5

  const float r = v4.x * kRed4.x + v4.y * kRed4.y + v4.z * kRed4.z + v4.w * kRed4.w +
                  v2.x * kRed2.x + v2.y * kRed2.y;
  const float g = v4.x * kGreen4.x + v4.y * kGreen4.y + v4.z * kGreen4.z + v4.w * kGreen4.w +
                  v2.x * kGreen2.x + v2.y * kGreen2.y;
  const float b = v4.x * kBlue4.x + v4.y * kBlue4.y + v4.z * kBlue4.z + v4.w * kBlue4.w +
                  v2.x * kBlue2.x + v2.y * kBlue2.y;
  return make_float3(saturate_f(r), saturate_f(g), saturate_f(b));
}

// The MATLAB/OpenCV jet ramp, as its three clipped triangles.
__device__ float3 jet(float x) {
  return make_float3(saturate_f(1.5f - fabsf(4.0f * x - 3.0f)),
                     saturate_f(1.5f - fabsf(4.0f * x - 2.0f)),
                     saturate_f(1.5f - fabsf(4.0f * x - 1.0f)));
}

__global__ void colorize_disparity_kernel(const float* __restrict__ disparity,
                                          const float* __restrict__ confidence, int width,
                                          int height, float min_disparity, float inv_span,
                                          float conf_threshold, int use_jet,
                                          cudaSurfaceObject_t surface) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) return;

  const int idx = y * width + x;
  const float d = disparity[idx];
  const bool usable = d > 0.0f && (confidence == nullptr || confidence[idx] >= conf_threshold);

  float3 rgb = make_float3(0.0f, 0.0f, 0.0f);
  if (usable) {
    const float t = saturate_f((d - min_disparity) * inv_span);
    rgb = use_jet ? jet(t) : turbo(t);
  }

  const uchar4 texel = make_uchar4(static_cast<unsigned char>(rgb.x * 255.0f + 0.5f),
                                   static_cast<unsigned char>(rgb.y * 255.0f + 0.5f),
                                   static_cast<unsigned char>(rgb.z * 255.0f + 0.5f), 255);
  surf2Dwrite(texel, surface, x * static_cast<int>(sizeof(uchar4)), y, cudaBoundaryModeClamp);
}

}  // namespace

void colorize_disparity(const float* disparity, const float* confidence, uint32_t width,
                        uint32_t height, float min_disparity, float max_disparity,
                        float conf_threshold, DisparityColormap colormap,
                        cudaSurfaceObject_t surface, cudaStream_t stream) {
  if (width == 0 || height == 0) return;

  // A collapsed or inverted range would divide by zero (or paint the image
  // backwards) once per pixel; one span of a pixel is the narrowest thing worth
  // drawing.
  const float span = fmaxf(max_disparity - min_disparity, 1.0f);

  const dim3 block(32, 8);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  colorize_disparity_kernel<<<grid, block, 0, stream>>>(
      disparity, confidence, static_cast<int>(width), static_cast<int>(height), min_disparity,
      1.0f / span, conf_threshold, colormap == DisparityColormap::Jet ? 1 : 0, surface);

  cuda_error_check(cudaGetLastError(), "colorize_disparity: kernel launch");
}

}  // namespace perception
