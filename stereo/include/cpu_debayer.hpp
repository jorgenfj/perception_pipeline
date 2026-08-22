#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Host-side demosaic, for looking at a frame on a machine with no GPU.
//
// This is not a second implementation of processing/src/transforms/debayer.cu
// and is not trying to be: that one is the pipeline's, it is bilinear, it runs
// on every frame, and its output feeds inference. This one exists so a person
// can see what the cameras are pointing at, and it is sized for that -- one
// output pixel per Bayer quad, no interpolation, on the CPU.
//
// Nothing here includes cuda_runtime.h, which is the point.

namespace perception {

// The Bayer orders a GigE camera actually emits, plus mono. Deliberately its
// own enum rather than pipeline/types.hpp's PixelFormat: that header includes
// cuda_runtime.h, and this half of the project does not have CUDA.
enum class HostPixelFormat : uint8_t {
  Mono8,
  BayerRG8,  // first row R G, second row G B
  BayerGR8,
  BayerGB8,
  BayerBG8,
};

// GenICam symbolic name -> enum. False for anything this viewer cannot draw,
// which the caller reports rather than guessing at. The name is the same string
// the manifest and the camera node map carry, so live and playback go through
// one mapping.
bool host_pixel_format_from_genicam(const std::string& name, HostPixelFormat& out);

const char* to_genicam_name(HostPixelFormat format);

// An RGB8 image, tightly packed, ready for glTexImage2D.
struct HostImage {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<unsigned char> rgb;  // width * height * 3

  void resize(uint32_t w, uint32_t h) {
    width = w;
    height = h;
    rgb.assign(static_cast<std::size_t>(w) * h * 3, 0);
  }
};

// Demosaic `src` into `out`.
//
// `decimate` is the side of the source block that becomes one output pixel and
// must be 2, 4 or 8 -- so the output is width/decimate by height/decimate. At 2
// that is exactly one Bayer quad per pixel: R from the red site, B from the
// blue, G averaged over the two greens, no interpolation and no invented
// detail. At 4 and 8 it is that plus a box average, which is what makes two
// 1440x1080 streams fit side by side in a window without the CPU noticing.
//
// 1 is not offered. A full-resolution host demosaic of a 60 Hz stereo pair is
// not something this tool should pretend it can do, and a block-replicated
// "full resolution" image would carry no more detail than decimate=2 while
// costing four times the memory.
void debayer_to_rgb(const unsigned char* src, uint32_t width, uint32_t height,
                    uint32_t stride_bytes, HostPixelFormat format, uint32_t decimate,
                    HostImage& out);

}  // namespace perception
