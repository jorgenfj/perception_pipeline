#include "ess_viewer_consumer.hpp"

#include <cstdio>

namespace perception {

struct EssViewerConsumer::Impl {};

EssViewerConsumer::EssViewerConsumer(RingPairConsumer&, uint32_t, const LatencyProbe&,
                                    const ImageDesc&, const utils::StereoCalibration&,
                                    const DisplayConfig&, const EssConfig&, int)
    : impl_(std::make_unique<Impl>()) {
  std::printf("ess viewer: disabled (this build has OPENGL_DISPLAY=OFF)\n");
}

EssViewerConsumer::~EssViewerConsumer() = default;

void EssViewerConsumer::start() {}
void EssViewerConsumer::stop() {}
bool EssViewerConsumer::closed() const { return false; }

}  // namespace perception
