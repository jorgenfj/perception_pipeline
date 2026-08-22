#pragma once

#include <cstdint>
#include <string>

#include "cpu_debayer.hpp"

struct GLFWwindow;

namespace perception {

// A window showing both cameras side by side, with the pair's skew under them.
//
// The whole path is host memory -> glTexSubImage2D -> screen. No CUDA-GL
// interop, no device pointers, nothing past OpenGL 1.1 -- so this runs on a
// laptop, on the Jetson, and on anything with a display and a GL driver, which
// is what visualization/gl_viewer.hpp deliberately cannot do (it registers the
// texture with CUDA, which is exactly why it is fast).
//
// Everything here must run on the thread that created the window: a GL context
// is thread-current.
class StereoView {
 public:
  struct Config {
    uint32_t window_width = 1600;
    uint32_t window_height = 700;
    bool vsync = true;  // a viewer, not a latency measurement: pace it to the panel

    // Full scale of the skew bar, either side of centre. Defaults to the
    // pairing tolerance, which is the number that decides whether a pair is a
    // pair at all, so the bar reads as "how close to not being one".
    uint64_t skew_scale_ns = 500'000;
  };

  // What to draw on top of the two frames. All of it is bars and blocks: a font
  // would be more code than the rest of this file, and for skew and staleness a
  // bar reads faster than digits anyway.
  struct Status {
    bool have[2] = {false, false};  // false dims that half and flags it stale
    int64_t skew_ns = 0;
    bool recording = false;
    bool paused = false;
  };

  // Throws if there is no display -- a headless box, or SSH with no forwarding.
  // The caller decides whether that is fatal; for this tool it usually is, and
  // for the recorder it is not.
  StereoView(const std::string& title, const Config& config);
  ~StereoView();

  StereoView(const StereoView&) = delete;
  StereoView& operator=(const StereoView&) = delete;

  // Pump events, sleeping up to `timeout_s`. Blocks on the display connection
  // rather than spinning, so an idle viewer costs nothing.
  void poll_wait(double timeout_s);

  bool should_close() const;

  // True once, per press, for keys the caller acts on: space toggles pause, r
  // toggles recording. Edge-detected here so the caller does not have to.
  bool take_pause_pressed();
  bool take_record_pressed();

  // Upload whatever is present and draw. Either side may be empty, which leaves
  // that half showing its previous frame -- dimmed, if Status says it is stale.
  void present(const HostImage& left, const HostImage& right, const Status& status);

  uint64_t presented() const { return presented_; }

 private:
  void upload(int side, const HostImage& image);
  void draw(const Status& status);
  bool take_edge(int key, bool& previous);

  Config config_;
  GLFWwindow* window_ = nullptr;

  unsigned int texture_[2] = {0, 0};
  uint32_t tex_width_[2] = {0, 0};
  uint32_t tex_height_[2] = {0, 0};

  bool pause_was_down_ = false;
  bool record_was_down_ = false;
  uint64_t presented_ = 0;
};

}  // namespace perception
