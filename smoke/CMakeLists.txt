# Toolchain smoke tests. These verify the build environment itself -- that CUDA
# and Spinnaker are found, compile, link, and load at runtime.

add_executable(cuda_smoke cuda_smoke.cu)
target_link_libraries(cuda_smoke PRIVATE CUDA::cudart)
add_test(NAME cuda_smoke COMMAND cuda_smoke)

if(PERCEPTION_WITH_SPINNAKER)
  add_executable(spinnaker_smoke spinnaker_smoke.cpp)
  target_link_libraries(spinnaker_smoke PRIVATE Spinnaker::Spinnaker)
  add_test(NAME spinnaker_smoke COMMAND spinnaker_smoke)
endif()

add_executable(yolo_smoke yolo_smoke.cpp)
target_link_libraries(yolo_smoke PRIVATE perception_core perception_processing perception_tensorrt)


target_compile_definitions(yolo_smoke PRIVATE
  YOLO_SMOKE_ENGINE_PATH="${CMAKE_SOURCE_DIR}/tensorrt/models/yolo26m_fp16.engine")
add_test(NAME yolo_smoke COMMAND yolo_smoke)
