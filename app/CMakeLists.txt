add_executable(acquire
  src/acquire_main.cpp
  src/app_config.cpp
  src/headless_yolo_consumer.cpp
  src/report.cpp)

target_include_directories(acquire PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(acquire PRIVATE
  perception_core
  perception_processing
  perception_camera
  perception_recording
  perception_tensorrt
  perception_visualization
  yaml-cpp)

if(PERCEPTION_SOURCE STREQUAL "spinnaker")
  target_sources(acquire PRIVATE src/source_spinnaker.cpp)
  target_link_libraries(acquire PRIVATE perception_spinnaker)
else()
  target_sources(acquire PRIVATE src/source_recording.cpp)
endif()

target_compile_definitions(acquire
  PRIVATE PERCEPTION_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config")

add_custom_command(TARGET acquire POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E rm -rf
          "$<TARGET_FILE_DIR:acquire>/models"
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          "${CMAKE_CURRENT_SOURCE_DIR}/../tensorrt/models"
          "$<TARGET_FILE_DIR:acquire>/models")
