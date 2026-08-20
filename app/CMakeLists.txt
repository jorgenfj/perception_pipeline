find_package(yaml-cpp REQUIRED)

add_executable(acquire
  src/acquire_main.cpp
  src/action_sync.cpp
  src/app_config.cpp
  src/headless_yolo_consumer.cpp
  src/report.cpp)

target_include_directories(acquire PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(acquire PRIVATE
  perception_core
  perception_processing
  perception_spinnaker
  perception_tensorrt
  perception_visualization
  yaml-cpp)
add_custom_command(TARGET acquire POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E rm -rf
          "$<TARGET_FILE_DIR:acquire>/config"
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          "${CMAKE_CURRENT_SOURCE_DIR}/config"
          "$<TARGET_FILE_DIR:acquire>/config")

add_custom_command(TARGET acquire POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E rm -rf
          "$<TARGET_FILE_DIR:acquire>/models"
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          "${CMAKE_CURRENT_SOURCE_DIR}/../tensorrt/models"
          "$<TARGET_FILE_DIR:acquire>/models")
