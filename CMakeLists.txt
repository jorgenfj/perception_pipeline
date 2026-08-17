cmake_minimum_required(VERSION 3.22)

project(perception_pipeline LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "87-real;120-real;120-virtual" CACHE STRING
      "CUDA architectures to build for")
endif()

option(PERCEPTION_WITH_SPINNAKER "Build against the FLIR Spinnaker SDK" OFF)
option(PERCEPTION_WITH_DISPLAY "Build the app's local GL viewer (needs GLFW and OpenGL)" ON)
option(PERCEPTION_BUILD_SMOKE "Build the toolchain smoke tests" ON)
option(PERCEPTION_BUILD_TESTS "Build the pipeline behaviour tests" ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

if(PERCEPTION_WITH_SPINNAKER)
  find_package(Spinnaker REQUIRED)
endif()

# Only the app uses these, but they are found here so a missing GLFW reports at
# configure time with the option to turn off, rather than midway through a build.
if(PERCEPTION_WITH_DISPLAY)
  # GLVND: on Jetson the NVIDIA driver installs libOpenGL/libGLX, and LEGACY
  # would resolve to a Mesa libGL that cannot share a context with CUDA.
  set(OpenGL_GL_PREFERENCE GLVND)
  find_package(OpenGL REQUIRED)
  find_package(glfw3 REQUIRED)
endif()

add_subdirectory(pipeline)

# spinnaker/ builds without CUDA and app/ is what joins the two, so the app
# only exists when both halves do.
if(PERCEPTION_WITH_SPINNAKER)
  add_subdirectory(spinnaker)
  add_subdirectory(app)
endif()

if(PERCEPTION_BUILD_SMOKE OR PERCEPTION_BUILD_TESTS)
  enable_testing()
endif()

if(PERCEPTION_BUILD_SMOKE)
  add_subdirectory(smoke)
endif()

# if(PERCEPTION_BUILD_TESTS)
#   add_subdirectory(tests)
# endif()

message(STATUS "CUDA toolkit ${CUDAToolkit_VERSION}, archs: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "GL viewer: ${PERCEPTION_WITH_DISPLAY}")
if(PERCEPTION_WITH_SPINNAKER)
  message(STATUS "Spinnaker ${Spinnaker_VERSION} (${Spinnaker_LIBRARY})")
endif()
