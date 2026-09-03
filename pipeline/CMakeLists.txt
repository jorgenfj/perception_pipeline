# The CUDA half of the project: rings, leases, and the stages between them.
# Knows nothing about cameras, config files, or any vendor SDK.

add_library(perception_core
  src/device_ring_buffer.cpp
  src/ring_lease.cpp
  src/host_ingress_ring.cpp
  src/device_transform.cpp
  src/transform_stage.cpp
  src/upload_stage.cpp
  src/download_stage.cpp
  src/host_frame_pool.cpp
  src/host_frame_ring.cpp
  src/latest_frame.cpp
  src/ring_pair_consumer.cpp
  )

target_include_directories(perception_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(perception_core PUBLIC CUDA::cudart Threads::Threads perception_utils)
