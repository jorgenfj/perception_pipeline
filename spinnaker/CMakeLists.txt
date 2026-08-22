add_library(perception_spinnaker
  src/action_sync.cpp
  src/spinnaker_source.cpp
  src/stereo_live.cpp)

target_include_directories(perception_spinnaker PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# PUBLIC on Spinnaker: spinnaker_source.hpp includes <Spinnaker.h> and takes a
# CameraPtr, so the SDK is part of this library's interface.
target_link_libraries(perception_spinnaker
  PUBLIC Spinnaker::Spinnaker perception_capture perception_camera)

# PUBLIC so the composition root's stats line is guarded by the same switch as
# the source's own tracing.
if(PERCEPTION_TRACE_POOL)
  target_compile_definitions(perception_spinnaker PUBLIC PERCEPTION_TRACE_POOL)
endif()
