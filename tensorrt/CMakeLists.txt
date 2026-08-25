find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
find_library(TENSORRT_NVINFER_LIB nvinfer)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIB)
  message(FATAL_ERROR
    "TensorRT (NvInfer.h / libnvinfer) not found -- install the TensorRT apt packages this "
    "Jetson image ships with")
endif()

# --- consistency guards ------------------------------------------------------
# We deliberately PIN BOTH TensorRT (major 10) and CUDA (major 12) so one build
# runs on both deployment targets:
#   * RTX 50-series desktop (Blackwell, sm_120) -- needs TRT >= 10.7 for sm_120
#   * Orin Jetson / JetPack 6  (Ampere,   sm_87) -- ships CUDA 12.x + TRT 10.x
# CUDA 12 + TRT 10 is the common denominator both platforms support
# Engines are also version-locked, so the major line must match the .engine file.
# The apt packages ship two flavours (+cuda12.x / +cuda13.x) -- always pick +cuda12.
set(PERCEPTION_TRT_MAJOR 10 CACHE STRING "Required TensorRT major version (must match the engine file)")
set(PERCEPTION_CUDA_MAJOR 12 CACHE STRING "Required CUDA toolkit major version")

# 1. CUDA toolkit is the expected major (CUDAToolkit_VERSION_* comes from the top-level find_package)
if(NOT CUDAToolkit_VERSION_MAJOR STREQUAL PERCEPTION_CUDA_MAJOR)
  message(FATAL_ERROR
    "CUDA ${PERCEPTION_CUDA_MAJOR}.x required, found ${CUDAToolkit_VERSION}. "
    "TensorRT must be the matching +cuda${PERCEPTION_CUDA_MAJOR}.x deb.")
endif()

# 2. TensorRT header major matches the engine's build
file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _trt_ver_hdr)
# Two spellings in the wild. TRT 10.7+ defines the numbers as TRT_*_ENTERPRISE and
# aliases NV_TENSORRT_* to them; the TRT 10.3 that JetPack 6 ships defines
# NV_TENSORRT_* as the literal. Try the alias source first, since where it exists
# NV_TENSORRT_MAJOR expands to a name rather than a digit and would not match.
string(REGEX MATCH "TRT_MAJOR_ENTERPRISE ([0-9]+)" _ "${_trt_ver_hdr}")
set(_trt_major "${CMAKE_MATCH_1}")
string(REGEX MATCH "TRT_MINOR_ENTERPRISE ([0-9]+)" _ "${_trt_ver_hdr}")
set(_trt_minor "${CMAKE_MATCH_1}")
if(_trt_major STREQUAL "")
  string(REGEX MATCH "NV_TENSORRT_MAJOR ([0-9]+)" _ "${_trt_ver_hdr}")
  set(_trt_major "${CMAKE_MATCH_1}")
  string(REGEX MATCH "NV_TENSORRT_MINOR ([0-9]+)" _ "${_trt_ver_hdr}")
  set(_trt_minor "${CMAKE_MATCH_1}")
endif()
if(_trt_major STREQUAL "")
  message(FATAL_ERROR
    "could not read a version out of ${TENSORRT_INCLUDE_DIR}/NvInferVersion.h -- it defines "
    "neither TRT_MAJOR_ENTERPRISE nor NV_TENSORRT_MAJOR as a number")
endif()
if(NOT _trt_major STREQUAL PERCEPTION_TRT_MAJOR)
  message(FATAL_ERROR
    "TensorRT ${PERCEPTION_TRT_MAJOR}.x required (engine compatibility), found ${_trt_major}.${_trt_minor} "
    "at ${TENSORRT_INCLUDE_DIR}. Install libnvinfer-dev=${PERCEPTION_TRT_MAJOR}.*+cuda${PERCEPTION_CUDA_MAJOR}.9.")
endif()
message(STATUS "TensorRT ${_trt_major}.${_trt_minor} (${TENSORRT_NVINFER_LIB})")

add_library(perception_tensorrt
  src/trt_engine.cpp
  src/yolo_engine.cpp
  src/draw_detections.cu
  )

target_include_directories(perception_tensorrt PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(perception_tensorrt PRIVATE "${TENSORRT_INCLUDE_DIR}")

target_link_libraries(perception_tensorrt PUBLIC perception_core perception_processing)
target_link_libraries(perception_tensorrt PRIVATE "${TENSORRT_NVINFER_LIB}")
