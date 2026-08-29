# CUDA image transforms that run on the pipeline's device rings. Depends on
# perception_core for the DeviceTransform interface and ring/geometry types,
# but the pipeline itself does not depend back on this -- transforms are
# consumers of the ring machinery, not part of it.

add_library(perception_processing
  src/transforms/debayer.cu
  src/transforms/rectification.cu
  src/transforms/rectify_map.cpp
  src/transforms/yolo_preprocess.cu
  )

target_include_directories(perception_processing PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# perception_capture is header-only and CUDA-free -- it is where the rig's
# calibration lives, and the rectification map is built straight from it. It
# carries perception_geometry, which is where the camera maths lives.
target_link_libraries(perception_processing PUBLIC perception_core perception_capture)
