#pragma once

namespace perception {

// One decoded box, already mapped out of model input space (e.g. 640x640)
// back into the source image's pixel coordinates -- the letterbox pad and
// scale are gone by the time a caller sees this.
struct Detection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

}  // namespace perception
