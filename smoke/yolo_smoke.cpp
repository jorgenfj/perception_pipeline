// Toolchain smoke test for the YOLO path: texture-read preprocess
// (processing/) -> TensorRT engine load and inference (tensorrt/), driven
// through YoloEngine's synchronous infer(). No camera and no ring-consumer
// threading needed -- one synthetic frame is pushed through the ring by hand
// and inferred on directly.
//
// The engine is a local build artifact (see app/config/acquire.yaml for how
// to make one) and is not checked in, so a machine without one skips rather
// than fails -- this is a toolchain check, not a guarantee the engine exists.

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "cuda_util.hpp"
#include "device_ring_buffer.hpp"
#include "types.hpp"
#include "yolo_engine.hpp"

using namespace perception;

int main() {
  std::FILE* f = std::fopen(YOLO_SMOKE_ENGINE_PATH, "rb");
  if (!f) {
    std::printf("yolo_smoke: no engine at %s -- skipping (see app/config/acquire.yaml)\n",
               YOLO_SMOKE_ENGINE_PATH);
    return 0;
  }
  std::fclose(f);

  const ImageDesc desc = packed_desc(640, 480, PixelFormat::RGBA8);

  DeviceRingTextureDesc tex;
  tex.width = desc.width;
  tex.height = desc.height;
  tex.pitch_bytes = desc.stride_bytes;

  DeviceRingBuffer ring(2, desc.bytes(), ReuseWait::DeviceWait, WritePolicy::RoundRobin,
                       /*max_consumers=*/1, /*device_id=*/0, tex);

  CudaStream stream;
  WriteLease write = ring.acquire_write(stream);
  cuda_error_check(cudaMemsetAsync(write.data(), 128, desc.bytes(), stream),
                   "yolo_smoke: cudaMemsetAsync");
  write.publish(1234);

  ReadLease lease = ring.lease_latest(0, stream);
  if (!lease.valid()) {
    std::fprintf(stderr, "yolo_smoke: could not lease the frame just published\n");
    return 1;
  }
  cuda_error_check(cudaStreamWaitEvent(stream, lease.data_ready_event(), 0),
                   "yolo_smoke: cudaStreamWaitEvent");

  YoloEngine::Config config;
  config.engine_path = YOLO_SMOKE_ENGINE_PATH;
  YoloEngine engine(desc, config);

  const std::vector<Detection> detections = engine.infer(lease.texture(), stream);
  lease.drop_hold();

  std::printf("yolo_smoke: OK, infer=%.2fms detections=%zu\n", engine.last_infer_ms(),
             detections.size());
  return 0;
}
