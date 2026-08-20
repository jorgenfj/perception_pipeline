#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>

#include "types.hpp"

struct GLFWwindow;

namespace perception {

// A debug window that shows a device RGBA8 ring slot without the frame ever
// leaving the GPU.
//
// The path is ring slot -> (D2D) -> a GL texture registered with CUDA -> screen.
// Registering the *texture* rather than a pixel buffer object is what keeps it
// to one copy: with a PBO you would still owe a glTexSubImage2D afterwards.
// This is also the payoff for the debayer emitting RGBA8 -- GL_RGBA8 is a
// format cudaGraphicsGLRegisterImage accepts, and a 3-channel texture is not.
//
// Everything here must run on the thread that created the window: the GL
// context is thread-current, and the CUDA-GL interop calls need it bound.
class GlViewer {
 public:
  struct Config {
    uint32_t window_width = 1280;
    uint32_t window_height = 960;
    // Off by default, and that default matters for what this tool is for:
    // vsync parks the swap until the panel is ready, adding up to a full
    // refresh period of latency to the very number being measured.
    bool vsync = false;
    // Full-scale of the latency bar. Frames past it clamp and read red.
    double latency_scale_ms = 100.0;
  };

  // Throws if there is no display to open -- a headless box, or an SSH session
  // with no X forwarding. Caller decides whether that is fatal.
  GlViewer(const ImageDesc& image, const Config& config);
  ~GlViewer();

  GlViewer(const GlViewer&) = delete;
  GlViewer& operator=(const GlViewer&) = delete;

  // Pump window events. Call every iteration, including idle ones, or the
  // window stops responding while the pipeline waits for a frame.
  void poll();

  // Pump window events, sleeping up to `timeout_s` for one to arrive. Blocks on
  // the display connection rather than spinning, so an idle viewer costs
  // nothing; the timeout caps how long it takes to notice a new ring frame,
  // which arrives on a different wakeup path entirely.
  void poll_wait(double timeout_s);

  bool should_close() const;

  // Enqueues the D2D copy out of `src` on `stream`, then draws. `src` must be
  // device RGBA8 matching the ImageDesc this was built with, and `stream` must
  // already be ordered behind the slot's readiness event.
  //
  // `latency_ms` drives the bar; pass a negative value while the probe has no
  // estimate yet and the bar is drawn empty.
  void present(const void* src, cudaStream_t stream, double latency_ms);

  // Same as present(), except `draw_boxes` -- if set -- is invoked with a
  // CUDA surface object bound to the same array the frame copy just wrote,
  // after that copy and before the array is unmapped, all on `stream`. That
  // is the hook for drawing detections without them ever crossing back to
  // the host: see YoloEngine::draw_into, which is what a caller passes here.
  // `draw_boxes` may be empty, meaning nothing extra is drawn.
  void present_gpu_boxes(const void* src, cudaStream_t stream, double latency_ms,
                        const std::function<void(cudaSurfaceObject_t)>& draw_boxes);

  // Frames presented, and the host time spent inside present() on the last one.
  uint64_t presented() const { return presented_; }
  double last_present_ms() const { return last_present_ms_; }

 private:
  void draw(double latency_ms);

  ImageDesc image_{};
  Config config_{};

  GLFWwindow* window_ = nullptr;
  unsigned int texture_ = 0;
  cudaGraphicsResource_t resource_ = nullptr;

  uint64_t presented_ = 0;
  double last_present_ms_ = 0.0;
};

}  // namespace perception
