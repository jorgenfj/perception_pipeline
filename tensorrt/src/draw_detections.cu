#include "draw_detections.hpp"

#include <cstddef>

#include "cuda_util.hpp"

namespace perception {
namespace {

constexpr int kBoxStride = 6;  // x1, y1, x2, y2, score, class_id
constexpr int kThickness = 2;

// --- label text ------------------------------------------------------------
//
// A 5x7 bitmap font covering exactly what a class name + confidence needs:
// space, '.', '0'-'9', 'A'-'Z'. Each row is a 5-bit literal written as the
// glyph's own pixels left-to-right (bit 4 = leftmost column), so a glyph can
// be proofread by eye against its shape rather than decoded from hex -- the
// usual way a small hand-authored bitmap font gets written in source.
constexpr int kGlyphWidth = 5;
constexpr int kGlyphHeight = 7;
constexpr int kGlyphCount = 38;
constexpr int kTextScale = 2;    // on-screen pixels per font pixel
constexpr int kGlyphGap = kTextScale;
constexpr int kChipPad = 2;
constexpr int kLabelMaxChars = 24;  // longest COCO name ("baseball glove") + " 0.00"

__constant__ uint8_t kFont[kGlyphCount][kGlyphHeight] = {
    // ' '
    {0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000},
    // '.'
    {0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100},
    // '0'-'9'
    {0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110},
    {0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110},
    {0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111},
    {0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110},
    {0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010},
    {0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110},
    {0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110},
    {0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000},
    {0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110},
    {0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100},
    // 'A'-'Z'
    {0b00100, 0b01010, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001},
    {0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110},
    {0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110},
    {0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110},
    {0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111},
    {0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000},
    {0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110},
    {0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001},
    {0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110},
    {0b00001, 0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b01110},
    {0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001},
    {0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111},
    {0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001},
    {0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001},
    {0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110},
    {0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000},
    {0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101},
    {0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001},
    {0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110},
    {0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100},
    {0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110},
    {0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100},
    {0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001},
    {0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b01010, 0b10001},
    {0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100},
    {0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111},
};

__device__ int glyph_index(char c) {
  if (c == '.') return 1;
  if (c >= '0' && c <= '9') return 2 + (c - '0');
  if (c >= 'A' && c <= 'Z') return 12 + (c - 'A');
  return 0;  // space, and anything unrecognised, is blank
}

// COCO's 80 classes, in the order every COCO-pretrained YOLO export uses --
// uppercase, since kFont has no lowercase. Confirmed against this project's
// own yolo26m_fp16.engine: class 0 read back as "person" and class 5 as
// "bus" against ultralytics/assets/bus.jpg (see the DLA comparison writeup),
// which only lines up if this ordering is right. A model trained on
// something other than COCO will just print the wrong name here, the same
// as any other hardcoded label set would.
__constant__ char kCocoNames[80][16] = {
    "PERSON",        "BICYCLE",      "CAR",           "MOTORCYCLE",    "AIRPLANE",
    "BUS",           "TRAIN",        "TRUCK",         "BOAT",          "TRAFFIC LIGHT",
    "FIRE HYDRANT",  "STOP SIGN",    "PARKING METER", "BENCH",         "BIRD",
    "CAT",           "DOG",          "HORSE",         "SHEEP",         "COW",
    "ELEPHANT",      "BEAR",         "ZEBRA",         "GIRAFFE",       "BACKPACK",
    "UMBRELLA",      "HANDBAG",      "TIE",           "SUITCASE",      "FRISBEE",
    "SKIS",          "SNOWBOARD",    "SPORTS BALL",   "KITE",          "BASEBALL BAT",
    "BASEBALL GLOVE", "SKATEBOARD",  "SURFBOARD",     "TENNIS RACKET", "BOTTLE",
    "WINE GLASS",    "CUP",          "FORK",          "KNIFE",         "SPOON",
    "BOWL",          "BANANA",       "APPLE",         "SANDWICH",      "ORANGE",
    "BROCCOLI",      "CARROT",       "HOT DOG",       "PIZZA",         "DONUT",
    "CAKE",          "CHAIR",        "COUCH",         "POTTED PLANT",  "BED",
    "DINING TABLE",  "TOILET",       "TV",            "LAPTOP",        "MOUSE",
    "REMOTE",        "KEYBOARD",     "CELL PHONE",    "MICROWAVE",     "OVEN",
    "TOASTER",       "SINK",         "REFRIGERATOR",  "BOOK",          "CLOCK",
    "VASE",          "SCISSORS",     "TEDDY BEAR",    "HAIR DRIER",    "TOOTHBRUSH",
};

// score in [0,1], so this is always "0.XX" -- three fixed characters after
// the class name, no general float formatting needed.
__device__ int format_label(int class_id, float score, char* out, int out_cap) {
  const char* name = (class_id >= 0 && class_id < 80) ? kCocoNames[class_id] : "?";
  int len = 0;
  for (int i = 0; name[i] != '\0' && len < out_cap - 1; ++i) out[len++] = name[i];
  if (len < out_cap - 1) out[len++] = ' ';

  const int hundredths = min(max(static_cast<int>(score * 100.0f + 0.5f), 0), 99);
  if (len < out_cap - 1) out[len++] = '0';
  if (len < out_cap - 1) out[len++] = '.';
  if (len < out_cap - 1) out[len++] = static_cast<char>('0' + hundredths / 10);
  if (len < out_cap - 1) out[len++] = static_cast<char>('0' + hundredths % 10);
  return len;
}

// --- color -------------------------------------------------------------

// Golden-ratio hue stepping: successive class ids land far apart around the
// hue wheel, which is what makes visually adjacent classes still read as
// different colors -- the standard trick for generating N deterministic,
// distinct colors without carrying a fixed palette table.
__device__ uchar4 class_color(int class_id) {
  const float hue = fmodf(static_cast<float>(class_id) * 0.61803398875f, 1.0f) * 6.0f;
  constexpr float s = 0.82f, v = 0.95f;
  const int hi = static_cast<int>(hue) % 6;
  const float f = hue - static_cast<int>(hue);
  const float p = v * (1.0f - s);
  const float q = v * (1.0f - s * f);
  const float t = v * (1.0f - s * (1.0f - f));
  float r, g, b;
  switch (hi) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    default: r = v; g = p; b = q; break;
  }
  return make_uchar4(static_cast<unsigned char>(r * 255.0f), static_cast<unsigned char>(g * 255.0f),
                     static_cast<unsigned char>(b * 255.0f), 255);
}

__device__ void put_pixel(cudaSurfaceObject_t surface, int x, int y, uint32_t img_w, uint32_t img_h,
                          uchar4 color) {
  if (x < 0 || x >= static_cast<int>(img_w) || y < 0 || y >= static_cast<int>(img_h)) return;
  surf2Dwrite(color, surface, x * static_cast<int>(sizeof(uchar4)), y, cudaBoundaryModeClamp);
}

__global__ void draw_detections_kernel(const float* __restrict__ output, uint32_t max_detections,
                                       float conf_threshold, float scale, float pad_x, float pad_y,
                                       uint32_t img_w, uint32_t img_h,
                                       cudaSurfaceObject_t surface) {
  // One block per candidate row: at most a few dozen boxes ever pass
  // conf_threshold out of up to a few hundred candidates, and every block
  // that does not returns immediately, so this costs nothing beyond one
  // score comparison per candidate.
  const uint32_t det = blockIdx.x;
  if (det >= max_detections) return;

  const float* row = output + static_cast<std::size_t>(det) * kBoxStride;
  const float score = row[4];
  if (score < conf_threshold) return;

  const float max_x = static_cast<float>(img_w - 1);
  const float max_y = static_cast<float>(img_h - 1);

  const int x1 = static_cast<int>(fminf(fmaxf((row[0] - pad_x) / scale, 0.0f), max_x));
  const int y1 = static_cast<int>(fminf(fmaxf((row[1] - pad_y) / scale, 0.0f), max_y));
  const int x2 = static_cast<int>(fminf(fmaxf((row[2] - pad_x) / scale, 0.0f), max_x));
  const int y2 = static_cast<int>(fminf(fmaxf((row[3] - pad_y) / scale, 0.0f), max_y));
  const int class_id = static_cast<int>(row[5]);
  const uchar4 color = class_color(class_id);

  // --- box outline, in the class's color ------------------------------
  const int w = max(x2 - x1, 0);
  const int h = max(y2 - y1, 0);
  const int perimeter = 2 * (w + h);

  // Every thread in the block walks a stride of this one box's perimeter --
  // a rectangle outline is a 1-D walk once unrolled edge by edge, so this
  // needs no 2-D indexing.
  for (int i = static_cast<int>(threadIdx.x); i < perimeter; i += static_cast<int>(blockDim.x)) {
    int px, py;
    if (i < w) {
      px = x1 + i;
      py = y1;  // top edge
    } else if (i < w + h) {
      px = x2;
      py = y1 + (i - w);  // right edge
    } else if (i < 2 * w + h) {
      px = x2 - (i - w - h);
      py = y2;  // bottom edge
    } else {
      px = x1;
      py = y2 - (i - 2 * w - h);  // left edge
    }
    for (int t = 0; t < kThickness; ++t) {
      put_pixel(surface, px + t, py + t, img_w, img_h, color);
    }
  }

  // The label chip below deliberately overlaps the box's top edge where it
  // sits inside it (see chip_y below); it has to be drawn after the outline
  // above so it wins that overlap, and __syncthreads() is what guarantees
  // every thread's outline writes actually landed first.
  __syncthreads();

  // --- label chip: class name + confidence, filled in the box's color --
  __shared__ char label[kLabelMaxChars];
  __shared__ int label_len;
  if (threadIdx.x == 0) {
    label_len = format_label(class_id, score, label, kLabelMaxChars);
  }
  __syncthreads();

  const int chip_w = label_len * (kGlyphWidth * kTextScale + kGlyphGap) + 2 * kChipPad;
  const int chip_h = kGlyphHeight * kTextScale + 2 * kChipPad;

  int chip_x = x1;
  int chip_y = y1 - chip_h;
  if (chip_y < 0) chip_y = y1;  // no room above the box -> hang the chip inside its top edge instead
  chip_x = min(chip_x, static_cast<int>(img_w) - chip_w);
  chip_x = max(chip_x, 0);

  // Fixed-luminance threshold, not a fixed text color: golden-ratio hues at
  // this saturation/value swing from very light (yellow-green) to very dark
  // (blue), and white-on-yellow or black-on-blue both read poorly.
  const float luma = 0.299f * color.x + 0.587f * color.y + 0.114f * color.z;
  const uchar4 text_color =
      luma > 140.0f ? make_uchar4(0, 0, 0, 255) : make_uchar4(255, 255, 255, 255);

  const int chip_pixels = chip_w * chip_h;
  for (int i = static_cast<int>(threadIdx.x); i < chip_pixels; i += static_cast<int>(blockDim.x)) {
    put_pixel(surface, chip_x + i % chip_w, chip_y + i / chip_w, img_w, img_h, color);
  }
  __syncthreads();  // text has to land after the chip background, not before it

  // One character per thread -- there are at most kLabelMaxChars of them,
  // so most of the block's 128 threads simply have nothing to do here.
  for (int c = static_cast<int>(threadIdx.x); c < label_len; c += static_cast<int>(blockDim.x)) {
    const int glyph = glyph_index(label[c]);
    const int gx = chip_x + kChipPad + c * (kGlyphWidth * kTextScale + kGlyphGap);
    const int gy = chip_y + kChipPad;
    for (int ry = 0; ry < kGlyphHeight; ++ry) {
      const uint8_t bits = kFont[glyph][ry];
      for (int rx = 0; rx < kGlyphWidth; ++rx) {
        if (!((bits >> (kGlyphWidth - 1 - rx)) & 1)) continue;
        for (int sy = 0; sy < kTextScale; ++sy) {
          for (int sx = 0; sx < kTextScale; ++sx) {
            put_pixel(surface, gx + rx * kTextScale + sx, gy + ry * kTextScale + sy, img_w, img_h,
                     text_color);
          }
        }
      }
    }
  }
}

}  // namespace

void draw_detections(const float* output, uint32_t max_detections, float conf_threshold,
                     float scale, float pad_x, float pad_y, uint32_t img_w, uint32_t img_h,
                     cudaSurfaceObject_t surface, cudaStream_t stream) {
  if (max_detections == 0) return;

  const dim3 grid(max_detections);
  const dim3 block(128);
  draw_detections_kernel<<<grid, block, 0, stream>>>(output, max_detections, conf_threshold, scale,
                                                     pad_x, pad_y, img_w, img_h, surface);

  cuda_error_check(cudaGetLastError(), "draw_detections: kernel launch");
}

}  // namespace perception
