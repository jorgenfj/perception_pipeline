# TensorRT engine loading and inference. Everything here assumes an engine
# file already exists -- built offline
find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
find_library(TENSORRT_NVINFER_LIB nvinfer)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIB)
  message(FATAL_ERROR
    "TensorRT (NvInfer.h / libnvinfer) not found -- install the TensorRT apt packages this "
    "Jetson image ships with")
endif()

add_library(perception_tensorrt
  src/trt_engine.cpp
  src/yolo_engine.cpp
  src/draw_detections.cu
  )

target_include_directories(perception_tensorrt PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(perception_tensorrt PRIVATE "${TENSORRT_INCLUDE_DIR}")

target_link_libraries(perception_tensorrt PUBLIC perception_core perception_processing)
target_link_libraries(perception_tensorrt PRIVATE "${TENSORRT_NVINFER_LIB}")
