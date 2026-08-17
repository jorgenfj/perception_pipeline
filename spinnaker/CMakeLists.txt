# The FLIR Spinnaker frame source, its config, and a standalone driver. All
# vendor coupling in the project stops here: swap this subproject for a replay
# or simulator source and nothing in perception_core changes.
#
# Deliberately CUDA-free. It links neither CUDA::cudart nor perception_core,
# and none of its headers include cuda_runtime.h -- frames leave through the
# FrameSink interface, which is expressed in void* and integers. That keeps
# this buildable and runnable on a machine with the SDK but no CUDA toolkit,
# which is what `spin_acquire` is for.
#
# The option still defaults to OFF because the SDK is not redistributable and
# has to be mounted into the build, not because this does not build.

# Owned here, not by the app: the camera schema is this subproject's, and the
# app parses its own sections from the same file rather than redefining these.
find_package(yaml-cpp REQUIRED)

add_library(perception_spinnaker
  src/camera_config.cpp
  src/heap_frame_sink.cpp
  src/spinnaker_source.cpp)

target_include_directories(perception_spinnaker PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# PUBLIC so the composition root's stats line is guarded by the same switch as
# the source's own tracing, rather than the two drifting apart.
if(PERCEPTION_TRACE_POOL)
  target_compile_definitions(perception_spinnaker PUBLIC PERCEPTION_TRACE_POOL)
endif()

# PUBLIC on Spinnaker: spinnaker_source.hpp includes <Spinnaker.h> and takes a
# CameraPtr in its constructor, so the SDK is part of this library's interface.
# yaml-cpp is PRIVATE -- the config headers expose plain structs, and only the
# .cpp ever sees a YAML::Node.
target_link_libraries(perception_spinnaker
  PUBLIC Spinnaker::Spinnaker
  PRIVATE yaml-cpp)

# Bring-up driver: camera straight into heap buffers, no GPU involved. Proves
# on the target that the transport accepts user-owned buffers before anything
# is built on top of that assumption.
add_executable(spin_acquire src/spin_acquire_main.cpp)
target_link_libraries(spin_acquire PRIVATE perception_spinnaker)

add_custom_command(TARGET spin_acquire POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/config"
          "$<TARGET_FILE_DIR:spin_acquire>/config")
