add_executable(ptp_trigger_test src/ptp_trigger_test_main.cpp)

target_link_libraries(ptp_trigger_test PRIVATE
  perception_stereo
  perception_spinnaker
  perception_camera
  yaml-cpp)

# Reuse stereo_view's config so a bare run finds the same stereo.yaml (serials,
# camera features, action_sync keys) the viewer uses.
target_compile_definitions(ptp_trigger_test
  PRIVATE PERCEPTION_CONFIG_DIR="${CMAKE_SOURCE_DIR}/tools/stereo_view/config")
