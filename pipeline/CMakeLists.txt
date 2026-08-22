# The CUDA half of the project: rings, leases, and the stages between them.
# Knows nothing about cameras, config files, or any vendor SDK.

add_library(perception_core
  src/device_ring_buffer.cpp
  src/ring_lease.cpp
  src/host_ingress_ring.cpp
  src/device_transform.cpp
  src/transform_stage.cpp
  src/upload_stage.cpp
  src/stereo_consumer.cpp
  )

target_include_directories(perception_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# perception_capture is capture/'s INTERFACE target: headers, no library, no
# dependencies of its own. The one thing needed from it is frame_pairing.hpp --
# the tolerance rule is a statement about the cameras' frame period rather than
# about the GPU, so it lives down there and the stereo consumer reaches out for
# it. No SDK and no yaml-cpp come with it, which is why it is that target and
# not perception_camera.
target_link_libraries(perception_core PUBLIC CUDA::cudart Threads::Threads perception_capture)
