find_package(yaml-cpp REQUIRED)

# INTERFACE and dependency-free: pipeline/ links this for frame_pairing.hpp
# alone and must not pick up yaml-cpp on the way.
add_library(perception_capture INTERFACE)
target_include_directories(perception_capture INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(perception_camera
  src/camera_config.cpp
  src/heap_frame_sink.cpp
  src/action_sync_check.cpp
  src/config_path.cpp)

target_link_libraries(perception_camera
  PUBLIC perception_capture Threads::Threads
  PRIVATE yaml-cpp)
