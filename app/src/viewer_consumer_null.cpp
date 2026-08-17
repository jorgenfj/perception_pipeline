#include "viewer_consumer.hpp"

namespace perception {

ViewerConsumer::ViewerConsumer(DeviceRingBuffer& ring, const LatencyProbe& probe,
                               const ImageDesc& desc, const DisplayConfig& config,
                               uint32_t consumer_id, int device_id)
    : ring_(&ring),
      probe_(&probe),
      desc_(desc),
      config_(config),
      consumer_id_(consumer_id),
      device_id_(device_id) {}

ViewerConsumer::~ViewerConsumer() = default;

void ViewerConsumer::start() {}
void ViewerConsumer::stop() {}
void ViewerConsumer::run() {}

}  // namespace perception
