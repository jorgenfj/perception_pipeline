// The host demosaic, on its own. Split out of recording_test.cpp, where it made
// a test of the file format pull in the viewer's pixel code to build.
//
// No GPU, no camera, no filesystem.
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "cpu_debayer.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++failures;
}

using perception::HostImage;
using perception::HostPixelFormat;

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 32;
constexpr std::size_t kFrameBytes = static_cast<std::size_t>(kWidth) * kHeight;

void test_debayer_round_trip() {
  // A frame that is flat red in the Bayer sense: R sites at max, G and B zero.
  // BayerRG8 puts R at (0,0), so those are the even row, even column bytes.
  std::vector<unsigned char> bayer(kFrameBytes, 0);
  for (uint32_t y = 0; y < kHeight; y += 2) {
    for (uint32_t x = 0; x < kWidth; x += 2) bayer[y * kWidth + x] = 255;
  }

  HostImage image;
  perception::debayer_to_rgb(bayer.data(), kWidth, kHeight, kWidth, HostPixelFormat::BayerRG8, 2,
                             image);
  check(image.width == kWidth / 2 && image.height == kHeight / 2,
        "decimate 2 gives one pixel per Bayer quad");
  check(image.rgb[0] == 255 && image.rgb[1] == 0 && image.rgb[2] == 0,
        "the red sites land in the red channel and nowhere else");

  // The same pattern read as BayerBG8 puts those sites in blue instead, which
  // is the failure mode of getting the pattern wrong, and it should be visible.
  perception::debayer_to_rgb(bayer.data(), kWidth, kHeight, kWidth, HostPixelFormat::BayerBG8, 2,
                            image);
  check(image.rgb[2] == 255 && image.rgb[0] == 0, "the pattern actually selects the channel");

  bool threw = false;
  try {
    perception::debayer_to_rgb(bayer.data(), kWidth, kHeight, kWidth, HostPixelFormat::BayerRG8, 1,
                              image);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "decimate 1 is refused rather than quietly replicating blocks");
}

}  // namespace

int main() {
  test_debayer_round_trip();

  std::printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
