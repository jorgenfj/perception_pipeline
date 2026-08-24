cmake_minimum_required(VERSION 3.22)

option(PERCEPTION_WITH_CUDA "Build the GPU pipeline (off leaves the CUDA-free tools)" ON)

if(PERCEPTION_WITH_CUDA)
  project(perception_pipeline LANGUAGES CXX CUDA)
else()
  project(perception_pipeline LANGUAGES CXX)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

if(PERCEPTION_WITH_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "87-real;120-real" CACHE STRING
        "CUDA architectures to build for")
  endif()
endif()

option(OPENGL_DISPLAY "Build the local GL viewers (needs GLFW and OpenGL)" ON)
option(PERCEPTION_TRACE_POOL "Log camera user-buffer pool starvation as it happens" OFF)
option(PERCEPTION_BUILD_SMOKE "Build the toolchain smoke tests" ON)
option(PERCEPTION_BUILD_TESTS "Build the pipeline behaviour tests" ON)

set(PERCEPTION_SOURCE "spinnaker" CACHE STRING "acquire's frame source: spinnaker | recording")
set_property(CACHE PERCEPTION_SOURCE PROPERTY STRINGS spinnaker recording)
if(NOT PERCEPTION_SOURCE MATCHES "^(spinnaker|recording)$")
  message(FATAL_ERROR "PERCEPTION_SOURCE must be 'spinnaker' or 'recording', got '${PERCEPTION_SOURCE}'")
endif()

if(PERCEPTION_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Spinnaker REQUIRED)

if(OPENGL_DISPLAY)
  # GLVND: on Jetson the NVIDIA driver installs libOpenGL/libGLX, and LEGACY
  # would resolve to a Mesa libGL that cannot share a context with CUDA.
  set(OpenGL_GL_PREFERENCE GLVND)
  find_package(OpenGL REQUIRED)
  find_package(glfw3 REQUIRED)
endif()

# --- the CUDA-free half -------------------------------------------------------
add_subdirectory(capture)
add_subdirectory(recording)
add_subdirectory(stereo)
add_subdirectory(spinnaker)
add_subdirectory(tools)

# --- the CUDA half ------------------------------------------------------------
if(PERCEPTION_WITH_CUDA)
  add_subdirectory(pipeline)
  add_subdirectory(processing)
  add_subdirectory(tensorrt)
  add_subdirectory(visualization)
  add_subdirectory(app)
endif()

if(PERCEPTION_BUILD_SMOKE OR PERCEPTION_BUILD_TESTS)
  enable_testing()
endif()

if(PERCEPTION_BUILD_SMOKE AND PERCEPTION_WITH_CUDA)
  add_subdirectory(smoke)
endif()

if(PERCEPTION_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(PERCEPTION_WITH_CUDA)
  message(STATUS "CUDA toolkit ${CUDAToolkit_VERSION}, archs: ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "acquire source: ${PERCEPTION_SOURCE}")
else()
  message(STATUS "CUDA: off -- building the CUDA-free tools only (spin_acquire, stereo_view)")
endif()
message(STATUS "Spinnaker ${Spinnaker_VERSION} (${Spinnaker_LIBRARY})")
message(STATUS "GL viewer: ${OPENGL_DISPLAY}")
message(STATUS "Pool trace: ${PERCEPTION_TRACE_POOL}")
