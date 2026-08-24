#include "yolo_viewer_consumer.hpp"

#include <cstdio>

namespace perception {

struct YoloViewerConsumer::Impl {};

YoloViewerConsumer::YoloViewerConsumer(DeviceRingBuffer&, const LatencyProbe&, const ImageDesc&,
                                      const DisplayConfig&, const YoloConfig&, uint32_t, int)
    : impl_(std::make_unique<Impl>()) {
  std::printf("yolo viewer: disabled (this build has OPENGL_DISPLAY=OFF)\n");
}

YoloViewerConsumer::~YoloViewerConsumer() = default;

void YoloViewerConsumer::start() {}
void YoloViewerConsumer::stop() {}
bool YoloViewerConsumer::closed() const { return false; }

}  // namespace perception
