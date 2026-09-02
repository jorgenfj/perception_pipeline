cmake_minimum_required(VERSION 3.22)

option(PERCEPTION_WITH_CUDA "Build the GPU pipeline (off leaves the CUDA-free tools)" ON)

# The architecture default has to be settled BEFORE project(), because enabling
# the CUDA language defines CMAKE_CUDA_ARCHITECTURES itself 
if(PERCEPTION_WITH_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(_wanted 87 120)

  set(_nvcc "${CMAKE_CUDA_COMPILER}")
  if(NOT _nvcc AND DEFINED ENV{CUDACXX})
    set(_nvcc "$ENV{CUDACXX}")
  endif()
  if(NOT _nvcc)
    find_program(_nvcc_program NAMES nvcc HINTS ENV CUDA_PATH PATH_SUFFIXES bin)
    set(_nvcc "${_nvcc_program}")
  endif()

  set(_supported "")
  if(_nvcc)
    execute_process(COMMAND "${_nvcc}" --list-gpu-arch
                    OUTPUT_VARIABLE _arch_list ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _arch_status)
    if(_arch_status EQUAL 0)
      string(REPLACE "\n" ";" _supported "${_arch_list}")
    endif()
  endif()

  set(_architectures "")
  foreach(_arch IN LISTS _wanted)
    if(NOT _supported OR "compute_${_arch}" IN_LIST _supported)
      list(APPEND _architectures "${_arch}-real")
    else()
      message(STATUS "CUDA: skipping sm_${_arch}, ${_nvcc} cannot emit it")
    endif()
  endforeach()
  if(NOT _architectures)
    message(FATAL_ERROR
      "none of the target architectures (${_wanted}) are supported by ${_nvcc} -- "
      "pass -DCMAKE_CUDA_ARCHITECTURES explicitly")
  endif()

  set(CMAKE_CUDA_ARCHITECTURES "${_architectures}" CACHE STRING
      "CUDA architectures to build for")
endif()

if(PERCEPTION_WITH_CUDA)
  project(perception_pipeline LANGUAGES CXX CUDA)
  message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
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
endif()

option(OPENGL_DISPLAY "Build the local GL viewers (needs GLFW and OpenGL)" ON)
option(PERCEPTION_TRACE_POOL "Log camera user-buffer pool starvation as it happens" OFF)
option(PERCEPTION_BUILD_SMOKE "Build the toolchain smoke tests" ON)
option(PERCEPTION_BUILD_TESTS "Build the pipeline behaviour tests" ON)
option(PERCEPTION_WITH_ROS "Build the ROS schema generator (needs a ROS 2 message tree)" OFF)

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

# MCAP dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED IMPORTED_TARGET GLOBAL liblz4)
pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET GLOBAL libzstd)

if(OPENGL_DISPLAY)
  # GLVND: on Jetson the NVIDIA driver installs libOpenGL/libGLX, and LEGACY
  # would resolve to a Mesa libGL that cannot share a context with CUDA.
  set(OpenGL_GL_PREFERENCE GLVND)
  find_package(OpenGL REQUIRED)
  find_package(glfw3 REQUIRED)
endif()

# Before any subdirectory that registers tests -- utils/ does, via
# gtest_discover_tests, and CTest only sees tests added after this point.
if(PERCEPTION_BUILD_SMOKE OR PERCEPTION_BUILD_TESTS)
  enable_testing()
endif()

# --- the CUDA-free half -------------------------------------------------------
# utils first: capture's calibration types are built out of it.
add_subdirectory(utils)
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
