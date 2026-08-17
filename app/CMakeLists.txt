# The composition root. This is the only place that depends on both halves: it
# owns the pipeline's config sections, the adapter that presents an ingress
# ring as a camera's FrameSink, and the vendor-name-to-pipeline-format mapping.
# The camera section is spinnaker's and is parsed by its loader.
#
# yaml-cpp comes in transitively through perception_spinnaker, which owns the
# camera schema; this only adds the pipeline's own sections on top.
find_package(yaml-cpp REQUIRED)

add_executable(acquire
  src/acquire_main.cpp
  src/app_config.cpp)

target_include_directories(acquire PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(acquire PRIVATE perception_core perception_spinnaker yaml-cpp)

# The viewer is a debug inspector, not part of the pipeline, so it is optional
# and its absence costs nothing: acquire_main compiles the window out entirely
# rather than carrying a runtime branch and a dead dependency.
#
# CUDA-GL interop needs a real GL context, which needs a display -- so this
# builds on the Jetson but not inside a plain CUDA container without one. The
# option lets that build turn it off rather than fail.
if(PERCEPTION_WITH_DISPLAY)
  # viewer_consumer.cpp is the display half of the pair; the headless build
  # links viewer_consumer_null.cpp instead, which is what keeps acquire_main
  # free of display #ifdefs rather than branching around the viewer.
  target_sources(acquire PRIVATE src/gl_viewer.cpp src/viewer_consumer.cpp)
  target_compile_definitions(acquire PRIVATE PERCEPTION_WITH_DISPLAY)
  # OpenGL::GL rather than the legacy variable: on Jetson the GLVND libGL is
  # what the NVIDIA driver installs, and the old FindOpenGL path picks the
  # wrong one.
  target_link_libraries(acquire PRIVATE glfw OpenGL::GL CUDA::cudart)
else()
  # Same interface, nothing behind it -- so nothing above the viewer has to
  # know which build this is.
  target_sources(acquire PRIVATE src/viewer_consumer_null.cpp)
endif()

# Next to the binary, so the default relative config path resolves when it is
# run straight out of the build tree.
add_custom_command(TARGET acquire POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/config"
          "$<TARGET_FILE_DIR:acquire>/config")
