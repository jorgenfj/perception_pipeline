# CUDA image transforms that run on the pipeline's device rings. Depends on
# perception_core for the DeviceTransform interface and ring/geometry types,
# but the pipeline itself does not depend back on this -- transforms are
# consumers of the ring machinery, not part of it.

add_library(perception_processing
  src/transforms/debayer.cu
  )

target_include_directories(perception_processing PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(perception_processing PUBLIC perception_core)
