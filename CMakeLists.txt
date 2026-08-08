cmake_minimum_required(VERSION 3.22)

project(perception_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# 87  = Jetson AGX Orin / Orin NX (Ampere)
# 120 = Blackwell, RTX 50-series (needs CUDA >= 12.8)
# On a dev machine with a GPU, pass the arch explicitly, e.g. -D...=89-real
# (`native` needs CMake >= 3.24, newer than what jammy provides).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "87-real;120-real;120-virtual" CACHE STRING
      "CUDA architectures to build for")
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(cuda_smoke src/main.cu)
target_link_libraries(cuda_smoke PRIVATE CUDA::cudart)

message(STATUS "CUDA toolkit ${CUDAToolkit_VERSION}, archs: ${CMAKE_CUDA_ARCHITECTURES}")
