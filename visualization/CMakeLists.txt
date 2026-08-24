add_library(perception_visualization)

target_include_directories(perception_visualization PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(perception_visualization PUBLIC perception_core perception_tensorrt)

if(OPENGL_DISPLAY)
  target_sources(perception_visualization PRIVATE
    src/gl_viewer.cpp src/viewer_consumer.cpp src/yolo_viewer_consumer.cpp)
  target_compile_definitions(perception_visualization PUBLIC OPENGL_DISPLAY)
  # OpenGL::GL rather than the legacy variable: on Jetson the GLVND libGL is
  # what the NVIDIA driver installs, and the old FindOpenGL path picks the
  # wrong one.
  target_link_libraries(perception_visualization PUBLIC glfw OpenGL::GL CUDA::cudart)
else()
  target_sources(perception_visualization PRIVATE
    src/viewer_consumer_null.cpp src/yolo_viewer_consumer_null.cpp)
endif()
