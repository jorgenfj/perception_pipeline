find_package(yaml-cpp REQUIRED)

add_library(perception_capture INTERFACE)
target_include_directories(perception_capture INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(perception_capture INTERFACE perception_utils)

add_library(perception_camera
  src/camera_config.cpp
  src/heap_frame_sink.cpp
  src/action_sync_check.cpp
  src/config_path.cpp
  src/stereo_calibration.cpp)

target_link_libraries(perception_camera
  PUBLIC perception_capture Threads::Threads
  PRIVATE yaml-cpp)
