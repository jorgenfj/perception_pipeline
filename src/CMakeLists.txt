add_library(perception_core
  device_ring_buffer.cpp
  ring_lease.cpp)

target_include_directories(perception_core PUBLIC "${PROJECT_SOURCE_DIR}/include")

target_link_libraries(perception_core PUBLIC CUDA::cudart Threads::Threads)
