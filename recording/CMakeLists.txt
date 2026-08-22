find_package(yaml-cpp REQUIRED)

add_library(perception_recording
  src/recording_reader.cpp
  src/recording_writer.cpp
  src/recording_source.cpp)

target_include_directories(perception_recording PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Threads is PUBLIC: the writer and the playback source both hold a std::thread
# in a header.
target_link_libraries(perception_recording
  PUBLIC perception_capture Threads::Threads
  PRIVATE yaml-cpp)
