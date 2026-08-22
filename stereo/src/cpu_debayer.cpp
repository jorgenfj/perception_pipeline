#include "cpu_debayer.hpp"

#include <stdexcept>

namespace perception {
namespace {

// Where R and B sit inside the 2x2 quad, as (row, col). The two greens are
// whatever is left, which is always the other diagonal.
struct QuadLayout {
  uint32_t r_row, r_col;
  uint32_t b_row, b_col;
};

QuadLayout layout_of(HostPixelFormat format) {
  switch (format) {
    case HostPixelFormat::BayerRG8:
      return {0, 0, 1, 1};
    case HostPixelFormat::BayerGR8:
      return {0, 1, 1, 0};
    case HostPixelFormat::BayerGB8:
      return {1, 0, 0, 1};
    case HostPixelFormat::BayerBG8:
      return {1, 1, 0, 0};
    case HostPixelFormat::Mono8:
      break;
  }
  return {0, 0, 1, 1};
}

}  // namespace

bool host_pixel_format_from_genicam(const std::string& name, HostPixelFormat& out) {
  if (name == "Mono8") {
    out = HostPixelFormat::Mono8;
    return true;
  }
  if (name == "BayerRG8") {
    out = HostPixelFormat::BayerRG8;
    return true;
  }
  if (name == "BayerGR8") {
    out = HostPixelFormat::BayerGR8;
    return true;
  }
  if (name == "BayerGB8") {
    out = HostPixelFormat::BayerGB8;
    return true;
  }
  if (name == "BayerBG8") {
    out = HostPixelFormat::BayerBG8;
    return true;
  }
  return false;
}

const char* to_genicam_name(HostPixelFormat format) {
  switch (format) {
    case HostPixelFormat::Mono8:
      return "Mono8";
    case HostPixelFormat::BayerRG8:
      return "BayerRG8";
    case HostPixelFormat::BayerGR8:
      return "BayerGR8";
    case HostPixelFormat::BayerGB8:
      return "BayerGB8";
    case HostPixelFormat::BayerBG8:
      return "BayerBG8";
  }
  return "?";
}

void debayer_to_rgb(const unsigned char* src, uint32_t width, uint32_t height,
                    uint32_t stride_bytes, HostPixelFormat format, uint32_t decimate,
                    HostImage& out) {
  if (decimate != 2 && decimate != 4 && decimate != 8) {
    throw std::runtime_error("debayer: decimate must be 2, 4 or 8");
  }
  if (src == nullptr || width < decimate || height < decimate) {
    throw std::runtime_error("debayer: image smaller than one output pixel");
  }

  const uint32_t out_w = width / decimate;
  const uint32_t out_h = height / decimate;
  out.resize(out_w, out_h);

  // Every output pixel averages (decimate/2)^2 Bayer quads. Holding the loop in
  // whole quads is what keeps the colour assignment a constant per pattern
  // rather than a per-pixel branch.
  const uint32_t quads = decimate / 2;
  const uint32_t quad_count = quads * quads;

  if (format == HostPixelFormat::Mono8) {
    for (uint32_t y = 0; y < out_h; ++y) {
      unsigned char* dst = out.rgb.data() + static_cast<std::size_t>(y) * out_w * 3;
      for (uint32_t x = 0; x < out_w; ++x) {
        uint32_t sum = 0;
        for (uint32_t dy = 0; dy < decimate; ++dy) {
          const unsigned char* row = src + static_cast<std::size_t>(y * decimate + dy) * stride_bytes;
          for (uint32_t dx = 0; dx < decimate; ++dx) sum += row[x * decimate + dx];
        }
        const unsigned char v = static_cast<unsigned char>(sum / (decimate * decimate));
        dst[x * 3 + 0] = v;
        dst[x * 3 + 1] = v;
        dst[x * 3 + 2] = v;
      }
    }
    return;
  }

  const QuadLayout quad = layout_of(format);
  // The greens are the sites the other two are not.
  const uint32_t g0_row = quad.r_row;
  const uint32_t g0_col = quad.b_col;
  const uint32_t g1_row = quad.b_row;
  const uint32_t g1_col = quad.r_col;

  for (uint32_t y = 0; y < out_h; ++y) {
    unsigned char* dst = out.rgb.data() + static_cast<std::size_t>(y) * out_w * 3;
    for (uint32_t x = 0; x < out_w; ++x) {
      uint32_t r = 0, g = 0, b = 0;
      for (uint32_t qy = 0; qy < quads; ++qy) {
        const uint32_t base_y = y * decimate + qy * 2;
        const uint32_t base_x = x * decimate;
        const unsigned char* row0 = src + static_cast<std::size_t>(base_y) * stride_bytes;
        const unsigned char* row1 = row0 + stride_bytes;
        for (uint32_t qx = 0; qx < quads; ++qx) {
          const uint32_t ox = base_x + qx * 2;
          const unsigned char* rows[2] = {row0, row1};
          r += rows[quad.r_row][ox + quad.r_col];
          b += rows[quad.b_row][ox + quad.b_col];
          // Averaged rather than picked: the two greens of a quad are the only
          // place this method has any redundancy at all, and using both halves
          // the green noise for free.
          g += (static_cast<uint32_t>(rows[g0_row][ox + g0_col]) +
                rows[g1_row][ox + g1_col] + 1) /
               2;
        }
      }
      dst[x * 3 + 0] = static_cast<unsigned char>(r / quad_count);
      dst[x * 3 + 1] = static_cast<unsigned char>(g / quad_count);
      dst[x * 3 + 2] = static_cast<unsigned char>(b / quad_count);
    }
  }
}

}  // namespace perception
