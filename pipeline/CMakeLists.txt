# The CUDA half of the project: rings, leases, and the stages between them.
# Knows nothing about cameras, config files, or any vendor SDK.

add_library(perception_core
  src/device_ring_buffer.cpp
  src/ring_lease.cpp
  src/host_ingress_ring.cpp
  src/device_transform.cpp
  src/transform_stage.cpp
  src/upload_stage.cpp
  )

target_include_directories(perception_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(perception_core PUBLIC CUDA::cudart Threads::Threads)
