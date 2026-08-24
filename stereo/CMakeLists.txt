find_package(yaml-cpp REQUIRED)

add_library(perception_stereo
  src/cpu_debayer.cpp
  src/stereo_config.cpp
  src/stereo_pairer.cpp)

target_include_directories(perception_stereo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(perception_stereo
  PUBLIC perception_capture perception_camera Threads::Threads
  PRIVATE yaml-cpp)

if(OPENGL_DISPLAY)
  target_sources(perception_stereo PRIVATE src/stereo_view.cpp)
  target_compile_definitions(perception_stereo PUBLIC OPENGL_DISPLAY)
  # Plain GLFW and OpenGL, no CUDA-GL interop: the frame reaches the texture
  # through glTexSubImage2D from host memory, which is the only way that works
  # without a GPU runtime.
  target_link_libraries(perception_stereo PUBLIC glfw OpenGL::GL)
endif()
