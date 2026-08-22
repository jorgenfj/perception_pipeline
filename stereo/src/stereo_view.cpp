#include "stereo_view.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace perception {
namespace {

// Nothing past GL 1.1 is used, so there is no GLEW, no glad and no runtime
// entry-point loading: every call below is exported by libGL directly. The same
// choice visualization/gl_viewer.cpp made, for the same reason.

void glfw_error(int code, const char* description) {
  std::fprintf(stderr, "glfw error %d: %s\n", code, description);
}

void quad(float x0, float y0, float x1, float y1) {
  glVertex2f(x0, y0);
  glVertex2f(x1, y0);
  glVertex2f(x1, y1);
  glVertex2f(x0, y1);
}

// Letterbox `image` inside the box, rather than stretching it: a distorted
// aspect makes the picture lie about the scene, which defeats using this to
// check the cameras at all.
//
// The box arrives in the 0..1 clip space the projection sets up, where one unit
// of x and one unit of y are not the same number of pixels -- so its aspect has
// to be worked out in framebuffer pixels. Doing it in clip space instead
// silently squashes both halves by the window's own aspect ratio.
void fit(float box_x0, float box_x1, float box_y0, float box_y1, uint32_t width, uint32_t height,
         int fb_w, int fb_h, float& x0, float& y0, float& x1, float& y1) {
  const double image_aspect = static_cast<double>(width) / std::max(1u, height);
  const double box_aspect = (static_cast<double>(box_x1 - box_x0) * fb_w) /
                            (static_cast<double>(box_y1 - box_y0) * fb_h);
  double w = 1.0, h = 1.0;
  if (box_aspect > image_aspect) {
    w = image_aspect / box_aspect;
  } else {
    h = box_aspect / image_aspect;
  }
  const float cx = (box_x0 + box_x1) * 0.5f;
  const float cy = (box_y0 + box_y1) * 0.5f;
  const float half_w = static_cast<float>(w) * (box_x1 - box_x0) * 0.5f;
  const float half_h = static_cast<float>(h) * (box_y1 - box_y0) * 0.5f;
  x0 = cx - half_w;
  x1 = cx + half_w;
  y0 = cy - half_h;
  y1 = cy + half_h;
}

}  // namespace

StereoView::StereoView(const std::string& title, const Config& config) : config_(config) {
  glfwSetErrorCallback(glfw_error);
  if (!glfwInit()) {
    throw std::runtime_error(
        "StereoView: glfwInit failed -- no display. Run on a machine with a screen, or use "
        "--no-display to record and pair headless");
  }

  // No profile hint, so this is a compatibility context and the fixed-function
  // calls below stay legal.
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

  window_ = glfwCreateWindow(static_cast<int>(config.window_width),
                             static_cast<int>(config.window_height), title.c_str(), nullptr,
                             nullptr);
  if (!window_) {
    glfwTerminate();
    throw std::runtime_error("StereoView: could not create a window");
  }

  glfwMakeContextCurrent(window_);
  glfwSwapInterval(config.vsync ? 1 : 0);

  // RGB rows of an odd width are not 4-byte aligned, and the default unpack
  // alignment of 4 would shear the image by a pixel per row.
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  glGenTextures(2, texture_);
  for (int side = 0; side < 2; ++side) {
    glBindTexture(GL_TEXTURE_2D, texture_[side]);
    // Linear down, nearest up: the window is usually smaller than the sensor,
    // but a magnified single-pixel defect should stay a defect rather than be
    // smoothed into something that looks like signal.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  }
  glBindTexture(GL_TEXTURE_2D, 0);
}

StereoView::~StereoView() {
  if (texture_[0]) glDeleteTextures(2, texture_);
  if (window_) glfwDestroyWindow(window_);
  glfwTerminate();
}

void StereoView::poll_wait(double timeout_s) { glfwWaitEventsTimeout(timeout_s); }

bool StereoView::should_close() const {
  return window_ == nullptr || glfwWindowShouldClose(window_) != 0 ||
         glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS ||
         glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS;
}

bool StereoView::take_edge(int key, bool& previous) {
  const bool down = window_ != nullptr && glfwGetKey(window_, key) == GLFW_PRESS;
  const bool pressed = down && !previous;
  previous = down;
  return pressed;
}

bool StereoView::take_pause_pressed() { return take_edge(GLFW_KEY_SPACE, pause_was_down_); }

bool StereoView::take_record_pressed() { return take_edge(GLFW_KEY_R, record_was_down_); }

void StereoView::upload(int side, const HostImage& image) {
  if (image.width == 0 || image.height == 0) return;

  glBindTexture(GL_TEXTURE_2D, texture_[side]);
  if (image.width != tex_width_[side] || image.height != tex_height_[side]) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, static_cast<GLsizei>(image.width),
                 static_cast<GLsizei>(image.height), 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    tex_width_[side] = image.width;
    tex_height_[side] = image.height;
  }
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(image.width),
                  static_cast<GLsizei>(image.height), GL_RGB, GL_UNSIGNED_BYTE, image.rgb.data());
  glBindTexture(GL_TEXTURE_2D, 0);
}

void StereoView::present(const HostImage& left, const HostImage& right, const Status& status) {
  upload(0, left);
  upload(1, right);
  draw(status);
  glfwSwapBuffers(window_);
  ++presented_;
}

void StereoView::draw(const Status& status) {
  int fb_w = 0, fb_h = 0;
  glfwGetFramebufferSize(window_, &fb_w, &fb_h);
  if (fb_w <= 0 || fb_h <= 0) return;

  glViewport(0, 0, fb_w, fb_h);
  glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  const float strip_h = std::min(0.10f, 34.0f / static_cast<float>(fb_h));
  const float gutter = 0.004f;

  glEnable(GL_TEXTURE_2D);
  for (int side = 0; side < 2; ++side) {
    if (tex_width_[side] == 0) continue;

    const float box_x0 = side == 0 ? 0.0f : 0.5f + gutter;
    const float box_x1 = side == 0 ? 0.5f - gutter : 1.0f;
    float x0, y0, x1, y1;
    fit(box_x0, box_x1, strip_h, 1.0f, tex_width_[side], tex_height_[side], fb_w, fb_h, x0, y0,
        x1, y1);

    glBindTexture(GL_TEXTURE_2D, texture_[side]);
    // A stale half is dimmed rather than blanked: the last frame is still the
    // most informative thing to show, it just must not be mistaken for a live
    // one.
    if (status.have[side]) {
      glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    } else {
      glColor4f(0.38f, 0.34f, 0.34f, 1.0f);
    }
    glBegin(GL_QUADS);
    // v flipped: GL's origin is bottom-left, the sensor's first row is the top.
    glTexCoord2f(0.0f, 1.0f); glVertex2f(x0, y0);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(x1, y0);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(x1, y1);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(x0, y1);
    glEnd();
  }
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);

  // --- status strip ---------------------------------------------------------
  const float pad = strip_h * 0.28f;
  const float bar_y0 = pad;
  const float bar_y1 = strip_h - pad;

  glBegin(GL_QUADS);
  glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
  quad(0.0f, 0.0f, 1.0f, strip_h);
  glEnd();

  // Recording indicator on the left of the strip, the width of the gutter.
  const float badge = strip_h * 0.5f;
  glBegin(GL_QUADS);
  if (status.recording) {
    glColor4f(0.90f, 0.20f, 0.20f, 1.0f);
  } else {
    glColor4f(0.20f, 0.20f, 0.24f, 1.0f);
  }
  quad(pad, (strip_h - badge) * 0.5f, pad + badge, (strip_h + badge) * 0.5f);
  if (status.paused) {
    // Two vertical bars, the universal pause glyph, next to the record badge.
    glColor4f(0.85f, 0.80f, 0.30f, 1.0f);
    const float x = pad * 2.0f + badge;
    quad(x, (strip_h - badge) * 0.5f, x + badge * 0.3f, (strip_h + badge) * 0.5f);
    quad(x + badge * 0.55f, (strip_h - badge) * 0.5f, x + badge * 0.85f,
         (strip_h + badge) * 0.5f);
  }
  glEnd();

  // --- skew bar -------------------------------------------------------------
  // Centre is zero skew, each end is the pairing tolerance. A pair that reads
  // near an end is one that nearly did not pair at all, which is the thing
  // worth seeing at a glance.
  const float track_x0 = 0.30f;
  const float track_x1 = 0.98f;
  const float centre = (track_x0 + track_x1) * 0.5f;

  glBegin(GL_QUADS);
  glColor4f(0.16f, 0.16f, 0.20f, 1.0f);
  quad(track_x0, bar_y0, track_x1, bar_y1);
  glEnd();

  // Ticks at +-1/2 and +-1 tolerance.
  glColor4f(0.45f, 0.45f, 0.52f, 1.0f);
  glBegin(GL_QUADS);
  for (int i = -2; i <= 2; ++i) {
    const float t = centre + (static_cast<float>(i) / 2.0f) * (track_x1 - track_x0) * 0.5f;
    const float h = (i == 0) ? 1.0f : 0.55f;
    quad(t - 0.0008f, bar_y0, t + 0.0008f, bar_y0 + (bar_y1 - bar_y0) * h);
  }
  glEnd();

  if (status.have[0] && status.have[1] && config_.skew_scale_ns > 0) {
    const double fraction =
        std::clamp(static_cast<double>(status.skew_ns) /
                       static_cast<double>(config_.skew_scale_ns),
                   -1.0, 1.0);
    const float x = centre + static_cast<float>(fraction) * (track_x1 - track_x0) * 0.5f;

    const double magnitude = std::fabs(fraction);
    if (magnitude < 0.34) {
      glColor4f(0.20f, 0.80f, 0.35f, 1.0f);
    } else if (magnitude < 0.67) {
      glColor4f(0.95f, 0.70f, 0.15f, 1.0f);
    } else {
      glColor4f(0.90f, 0.25f, 0.25f, 1.0f);
    }
    glBegin(GL_QUADS);
    quad(std::min(centre, x), bar_y0 + (bar_y1 - bar_y0) * 0.25f, std::max(centre, x),
         bar_y1 - (bar_y1 - bar_y0) * 0.25f);
    quad(x - 0.003f, bar_y0, x + 0.003f, bar_y1);
    glEnd();
  } else {
    // Unpaired: no skew exists to draw, and drawing zero would be a lie.
    glColor4f(0.55f, 0.25f, 0.25f, 1.0f);
    glBegin(GL_QUADS);
    quad(centre - 0.003f, bar_y0, centre + 0.003f, bar_y1);
    glEnd();
  }
}

}  // namespace perception
