#include "gl_viewer.hpp"

#include <GLFW/glfw3.h>
// cuda_gl_interop.h needs the GL types, so it follows the GL header rather than
// sorting with the other CUDA includes.
#include <cuda_gl_interop.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "cuda_util.hpp"
#include "latency_probe.hpp"

namespace perception {
namespace {

// Deliberately nothing past GL 1.1 is used. Registering the texture instead of
// a pixel buffer removes the only reason this would have needed glGenBuffers
// and friends, which in turn removes any need for GLEW, glad, or runtime
// entry-point loading -- everything below is exported by libGL directly.

void glfw_error(int code, const char* description) {
  std::fprintf(stderr, "glfw error %d: %s\n", code, description);
}

int glfw_refcount = 0;

void quad(float x0, float y0, float x1, float y1) {
  glVertex2f(x0, y0);
  glVertex2f(x1, y0);
  glVertex2f(x1, y1);
  glVertex2f(x0, y1);
}

}  // namespace

GlViewer::GlViewer(const ImageDesc& image, const Config& config)
    : image_(image), config_(config) {
  if (image.format != PixelFormat::RGBA8) {
    throw std::runtime_error("GlViewer: only RGBA8 is displayable");
  }
  if (image.width == 0 || image.height == 0) {
    throw std::runtime_error("GlViewer: zero-sized image");
  }

  glfwSetErrorCallback(glfw_error);
  if (glfw_refcount == 0 && !glfwInit()) {
    throw std::runtime_error(
        "GlViewer: glfwInit failed -- no display. Run on the Jetson's own screen, or set "
        "display.enable to false to run headless");
  }
  ++glfw_refcount;

  // No profile hint, so this asks for a compatibility context and the
  // fixed-function calls below stay legal. A debug viewer is not worth a
  // shader pipeline.
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

  window_ = glfwCreateWindow(static_cast<int>(config.window_width),
                             static_cast<int>(config.window_height), "perception :: acquire",
                             nullptr, nullptr);
  if (!window_) {
    if (--glfw_refcount == 0) glfwTerminate();
    throw std::runtime_error("GlViewer: could not create a window");
  }

  glfwMakeContextCurrent(window_);
  glfwSwapInterval(config.vsync ? 1 : 0);

  glGenTextures(1, &texture_);
  glBindTexture(GL_TEXTURE_2D, texture_);
  // Linear on minification because the window is usually smaller than the
  // sensor; nearest on magnification so single-pixel defects stay visible
  // rather than being smoothed into something that looks like signal.
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(image.width),
               static_cast<GLsizei>(image.height), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glBindTexture(GL_TEXTURE_2D, 0);

  // WriteDiscard: every frame overwrites the whole texture, so the driver need
  // not preserve or fetch back the previous contents.
  const cudaError_t err = cudaGraphicsGLRegisterImage(
      &resource_, texture_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
  if (err != cudaSuccess) {
    glDeleteTextures(1, &texture_);
    glfwDestroyWindow(window_);
    if (--glfw_refcount == 0) glfwTerminate();
    throw std::runtime_error(std::string("GlViewer: cudaGraphicsGLRegisterImage: ") +
                             cudaGetErrorString(err) +
                             " -- the GL context and the CUDA device must be the same GPU");
  }
}

GlViewer::~GlViewer() {
  if (resource_) cudaGraphicsUnregisterResource(resource_);
  if (texture_) glDeleteTextures(1, &texture_);
  if (window_) glfwDestroyWindow(window_);
  if (--glfw_refcount == 0) glfwTerminate();
}

void GlViewer::poll() { glfwPollEvents(); }

void GlViewer::poll_wait(double timeout_s) { glfwWaitEventsTimeout(timeout_s); }

bool GlViewer::should_close() const {
  return window_ == nullptr || glfwWindowShouldClose(window_) ||
         glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS ||
         glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS;
}

void GlViewer::present(const void* src, cudaStream_t stream, double latency_ms) {
  present_gpu_boxes(src, stream, latency_ms, nullptr);
}

void GlViewer::present_gpu_boxes(const void* src, cudaStream_t stream, double latency_ms,
                                 const std::function<void(cudaSurfaceObject_t)>& draw_boxes) {
  const uint64_t started = LatencyProbe::host_now_ns();

  cudaArray_t array = nullptr;
  cuda_error_check(cudaGraphicsMapResources(1, &resource_, stream),
                   "GlViewer: cudaGraphicsMapResources");
  cuda_error_check(cudaGraphicsSubResourceGetMappedArray(&array, resource_, 0, 0),
                   "GlViewer: cudaGraphicsSubResourceGetMappedArray");

  cuda_error_check(cudaMemcpy2DToArrayAsync(array, 0, 0, src, image_.stride_bytes,
                                            static_cast<size_t>(image_.width) * 4, image_.height,
                                            cudaMemcpyDeviceToDevice, stream),
                   "GlViewer: cudaMemcpy2DToArrayAsync");

  if (draw_boxes) {
    // Bound to the same array the frame copy above just wrote, so whatever
    // draw_boxes enqueues here lands on top of this frame specifically --
    // not a stale one -- because it is stream-ordered after that copy.
    cudaResourceDesc res_desc{};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;
    cudaSurfaceObject_t surface = 0;
    cuda_error_check(cudaCreateSurfaceObject(&surface, &res_desc),
                     "GlViewer: cudaCreateSurfaceObject");
    draw_boxes(surface);
    // Safe to destroy right away even though the kernel it was passed to is
    // still queued: the object is a lightweight binding resolved at launch,
    // not something the in-flight kernel dereferences later, and the array
    // it points at is not freed until the unmap below (which is also on
    // this stream, so still ordered after that kernel).
    cuda_error_check(cudaDestroySurfaceObject(surface), "GlViewer: cudaDestroySurfaceObject");
  }

  // Unmapping on the same stream is what orders the copy (and any box
  // drawing above) ahead of GL's read of the texture. Without the stream
  // argument that guarantee is against the default stream instead, and the
  // quad below can sample a half-written frame.
  cuda_error_check(cudaGraphicsUnmapResources(1, &resource_, stream),
                   "GlViewer: cudaGraphicsUnmapResources");

  draw(latency_ms);
  glfwSwapBuffers(window_);

  ++presented_;
  last_present_ms_ = static_cast<double>(LatencyProbe::host_now_ns() - started) * 1e-6;
}

void GlViewer::draw(double latency_ms) {
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

  // Letterbox rather than stretch: a distorted aspect makes the picture lie
  // about the scene, which defeats using this to check the camera at all.
  const double image_aspect = static_cast<double>(image_.width) / image_.height;
  const double window_aspect = static_cast<double>(fb_w) / fb_h;
  double w = 1.0, h = 1.0;
  if (window_aspect > image_aspect) {
    w = image_aspect / window_aspect;
  } else {
    h = window_aspect / image_aspect;
  }
  const float x0 = static_cast<float>((1.0 - w) * 0.5);
  const float y0 = static_cast<float>((1.0 - h) * 0.5);
  const float x1 = static_cast<float>(x0 + w);
  const float y1 = static_cast<float>(y0 + h);

  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  glBegin(GL_QUADS);
  // v flipped: GL's origin is bottom-left, the sensor's first row is the top.
  glTexCoord2f(0.0f, 1.0f); glVertex2f(x0, y0);
  glTexCoord2f(1.0f, 1.0f); glVertex2f(x1, y0);
  glTexCoord2f(1.0f, 0.0f); glVertex2f(x1, y1);
  glTexCoord2f(0.0f, 0.0f); glVertex2f(x0, y1);
  glEnd();
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);

  // Detection boxes, if any, are already burned into the frame texture
  // itself by the time this runs -- see GlViewer::present_gpu_boxes -- so
  // there is nothing more to draw for them here.

  // --- latency bar ---------------------------------------------------------
  // No text: a font would be more code than the rest of this file. A bar
  // against fixed ticks reads faster than digits anyway when the thing you are
  // looking for is jitter.
  const float bar_h = std::min(0.06f, 28.0f / static_cast<float>(fb_h));
  const float pad = bar_h * 0.35f;

  glBegin(GL_QUADS);
  glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
  quad(0.0f, 0.0f, 1.0f, bar_h + pad * 2.0f);
  glColor4f(0.18f, 0.18f, 0.22f, 1.0f);
  quad(pad, pad, 1.0f - pad, pad + bar_h);
  glEnd();

  if (latency_ms >= 0.0) {
    const double fraction = std::clamp(latency_ms / config_.latency_scale_ms, 0.0, 1.0);
    if (fraction < 0.34) {
      glColor4f(0.20f, 0.80f, 0.35f, 1.0f);
    } else if (fraction < 0.67) {
      glColor4f(0.95f, 0.70f, 0.15f, 1.0f);
    } else {
      glColor4f(0.90f, 0.25f, 0.25f, 1.0f);
    }
    glBegin(GL_QUADS);
    quad(pad, pad, pad + static_cast<float>(fraction) * (1.0f - pad * 2.0f), pad + bar_h);
    glEnd();
  }

  // A tick every 10 ms of the configured scale, so the bar has a readable unit.
  const int ticks = std::max(1, static_cast<int>(config_.latency_scale_ms / 10.0));
  glColor4f(0.55f, 0.55f, 0.60f, 1.0f);
  glBegin(GL_QUADS);
  for (int i = 1; i < ticks; ++i) {
    const float t = pad + (static_cast<float>(i) / ticks) * (1.0f - pad * 2.0f);
    quad(t, pad, t + 0.0012f, pad + bar_h * ((i % 5 == 0) ? 1.0f : 0.45f));
  }
  glEnd();
}

}  // namespace perception
