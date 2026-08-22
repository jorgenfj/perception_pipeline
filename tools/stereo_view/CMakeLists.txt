add_executable(stereo_view src/stereo_view_main.cpp)

target_link_libraries(stereo_view PRIVATE
  perception_stereo
  perception_recording
  perception_spinnaker
  perception_camera
  yaml-cpp)

target_compile_definitions(stereo_view
  PRIVATE PERCEPTION_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config")
