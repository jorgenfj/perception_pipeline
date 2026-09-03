#pragma once

#include <cstddef>
#include <cstdint>

namespace perception {

// One frame of host-side pixels, handed out by HostFramePool.
//
// Deliberately free of <cuda_runtime.h>: this is what crosses the boundary to
// sinks, and a recorder or a publisher should not have to link CUDA to describe
// what it was given. The pool that allocates these is the CUDA-facing half.
struct HostFrame {
  // Pinned. Valid for as long as any shared_ptr to this frame is alive.
  void* data = nullptr;
  std::size_t bytes = 0;

  // The camera clock of the frame this was derived from, verbatim. For a
  // disparity map that is the reference eye's stamp, read before the lease was
  // dropped -- it is what correlates this frame with the recorded images, and
  // the only field a reader needs to line the two up.
  uint64_t timestamp_ns = 0;

  // CLOCK_REALTIME when the frame was published, i.e. once its copy had
  // retired. Diagnostic: against timestamp_ns this is end-to-end
  // exposure-to-host latency, not merely transport.
  uint64_t host_ready_ns = 0;

  // Monotonic per pool, never reused. A gap means the producer dropped a frame
  // rather than a sink losing one, which is the distinction worth keeping.
  uint64_t sequence = 0;

  uint32_t width = 0;
  uint32_t height = 0;
};

}  // namespace perception
