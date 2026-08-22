add_executable(spin_acquire src/spin_acquire_main.cpp)

target_link_libraries(spin_acquire PRIVATE
  perception_spinnaker
  perception_recording
  perception_camera)

target_compile_definitions(spin_acquire
  PRIVATE PERCEPTION_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config")
