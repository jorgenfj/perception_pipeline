#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "types.hpp"

namespace perception {

// Applying a rectification lookup map on the GPU.
//
// Deliberately knows nothing about cameras. The map is just a table of source
// coordinates by the time it gets here; building it is camera maths and lives
// in transforms/rectify_map.hpp, which is host-only and pulls in Eigen. Keeping
// that out of this header is what keeps nvcc from having to compile Eigen for
// every translation unit that wants to run the kernel.
//
// The table is float2 rather than the fixed-point (short2, ushort) pair OpenCV
// uses for CV_16SC2 maps. 8 bytes per pixel is 12MB for a 1440x1080 pair, which
// is nothing against a GPU's bandwidth, and it hands the coordinate straight to
// the TMU in the form the TMU already wants -- the fixed-point form would have
// to be unpacked and rescaled per pixel, spending ALU to save bandwidth that is
// not the bottleneck here.

// Which coordinate convention the map entries are written in. This has to match
// the `normalized_coords` of the texture that will be sampled with them -- a
// mismatch is not a visible error, it collapses the whole image into the top
// left texel or scatters it far outside, so it is stated at build time rather
// than assumed.
enum class RectifyMapCoords {
  // [0, width) x [0, height), texel centres on the half-integers. Pair with
  // DeviceRingTextureDesc::normalized_coords = false.
  Pixels,
  // The same, divided through by the image size, so [0, 1). This is the ring's
  // default and therefore the default here.
  Normalized,
};

// How many of a map's entries land outside the source image. Those pixels are
// whatever the texture's address mode says -- edge replication under the ring's
// default cudaAddressModeClamp -- so a large count means a lot of the rectified
// frame is smeared border rather than image, and the caller wants to know
// before it feeds that to a matcher. A stereoRectify run at alpha=0 reports a
// full-frame valid ROI, for which the expected count is 0.
std::size_t count_out_of_frame(const std::vector<float2>& map, uint32_t width, uint32_t height,
                               RectifyMapCoords coords);

// The map, resident on the device, plus the kernel that applies it.
//
// Owns one device allocation for the lifetime of the object. Build it once per
// eye at startup: the constructor allocates and does a blocking upload, and is
// not something to call on the frame path.
//
// Takes an already-built map rather than a calibration, which is what keeps
// this class free of the camera maths. make_rectify_transform() in
// transforms/rectify_map.hpp is the one-line way to go straight from a
// CameraCalibration to one of these.
class RectifyTransform {
 public:
  // `map` must be packed row-major, width * height entries, in `coords`.
  RectifyTransform(const std::vector<float2>& map, uint32_t width, uint32_t height,
                   RectifyMapCoords coords = RectifyMapCoords::Normalized);
  ~RectifyTransform();

  RectifyTransform(const RectifyTransform&) = delete;
  RectifyTransform& operator=(const RectifyTransform&) = delete;

  // Rectified geometry equals calibrated geometry, packed RGBA8.
  ImageDesc output_desc() const { return packed_desc(width_, height_, PixelFormat::RGBA8); }

  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }
  RectifyMapCoords coords() const { return coords_; }
  std::size_t out_of_frame() const { return out_of_frame_; }

  // `src_texture` must be an RGBA8 texture read as cudaReadModeNormalizedFloat
  // with cudaFilterModeLinear -- the ring's defaults. Element-type reads cannot
  // be filtered at all, and unfiltered remap is nearest-neighbour, which throws
  // away most of what rectification is for.
  //
  // `output` must match output_desc() except for a longer stride.
  void enqueue(cudaTextureObject_t src_texture, void* dst, const ImageDesc& output,
               cudaStream_t stream) const;

  const char* name() const { return "RectifyLookup"; }

 private:
  float2* map_ = nullptr;  // device, width_ * height_ entries, packed
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  RectifyMapCoords coords_ = RectifyMapCoords::Normalized;
  std::size_t out_of_frame_ = 0;
};

}  // namespace perception
