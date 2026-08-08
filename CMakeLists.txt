cmake_minimum_required(VERSION 3.22)

project(perception_pipeline LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Collect executables in build/bin rather than mirroring the source tree.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "87-real;120-real;120-virtual" CACHE STRING
      "CUDA architectures to build for")
endif()

option(PERCEPTION_WITH_SPINNAKER "Build against the FLIR Spinnaker SDK" ON)
option(PERCEPTION_BUILD_SMOKE "Build the toolchain smoke tests" ON)

find_package(CUDAToolkit REQUIRED)

if(PERCEPTION_WITH_SPINNAKER)
  find_package(Spinnaker REQUIRED)
endif()

# enable_testing() must be called here, not in smoke/ -- ctest only discovers
# tests registered from the directory the build tree is rooted at.
if(PERCEPTION_BUILD_SMOKE)
  enable_testing()
  add_subdirectory(smoke)
endif()

message(STATUS "CUDA toolkit ${CUDAToolkit_VERSION}, archs: ${CMAKE_CUDA_ARCHITECTURES}")
if(PERCEPTION_WITH_SPINNAKER)
  message(STATUS "Spinnaker ${Spinnaker_VERSION} (${Spinnaker_LIBRARY})")
endif()
