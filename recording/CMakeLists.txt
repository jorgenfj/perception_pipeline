find_package(yaml-cpp REQUIRED)

add_library(perception_recording
  src/mcap_player.cpp
  src/image_replay_source.cpp
  src/mcap_recorder.cpp
  src/ros_messages.cpp
  src/recording_config.cpp
  src/replay_config.cpp)

target_include_directories(perception_recording PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(perception_recording PRIVATE "${CMAKE_SOURCE_DIR}/third_party")

target_link_libraries(perception_recording
  PUBLIC perception_capture perception_utils Threads::Threads
  PRIVATE yaml-cpp PkgConfig::LZ4 PkgConfig::ZSTD)
