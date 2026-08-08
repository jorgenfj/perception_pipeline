# Toolchain smoke tests. These verify the build environment itself -- that CUDA
# and Spinnaker are found, compile, link, and load at runtime. They are not unit
# tests of pipeline behaviour.

add_executable(cuda_smoke cuda_smoke.cu)
target_link_libraries(cuda_smoke PRIVATE CUDA::cudart)
add_test(NAME cuda_smoke COMMAND cuda_smoke)

if(PERCEPTION_WITH_SPINNAKER)
  add_executable(spinnaker_smoke spinnaker_smoke.cpp)
  target_link_libraries(spinnaker_smoke PRIVATE Spinnaker::Spinnaker)
  add_test(NAME spinnaker_smoke COMMAND spinnaker_smoke)
endif()
