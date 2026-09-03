# Behaviour tests for pipeline code, as opposed to smoke/, which only checks
# that the toolchain works.

# --- no GPU, no camera, no fixtures -------------------------------------------
# These build and run in a -DPERCEPTION_WITH_CUDA=OFF configuration, which is
# most of the point of the code they cover.
add_executable(frame_pairing_test frame_pairing_test.cpp)
target_link_libraries(frame_pairing_test PRIVATE perception_capture)
add_test(NAME frame_pairing_test COMMAND frame_pairing_test)

add_executable(stereo_pairer_test stereo_pairer_test.cpp)
target_link_libraries(stereo_pairer_test PRIVATE perception_stereo)
add_test(NAME stereo_pairer_test COMMAND stereo_pairer_test)

add_executable(topic_graph_test topic_graph_test.cpp "${CMAKE_SOURCE_DIR}/app/src/topic_graph.cpp")
target_link_libraries(topic_graph_test PRIVATE perception_core)
target_include_directories(topic_graph_test PRIVATE "${CMAKE_SOURCE_DIR}/app/include")
add_test(NAME topic_graph_test COMMAND topic_graph_test)

add_executable(cdr_writer_test cdr_writer_test.cpp)
target_link_libraries(cdr_writer_test PRIVATE perception_recording)
add_test(NAME cdr_writer_test COMMAND cdr_writer_test)

# The round trip, so it needs the camera library for HeapFrameSink as well.
add_executable(mcap_player_test mcap_player_test.cpp)
target_link_libraries(mcap_player_test
  PRIVATE perception_recording perception_camera PkgConfig::LZ4 PkgConfig::ZSTD)
target_include_directories(mcap_player_test PRIVATE "${CMAKE_SOURCE_DIR}/third_party")
add_test(NAME mcap_player_test COMMAND mcap_player_test)

add_executable(mcap_recorder_test mcap_recorder_test.cpp)
target_link_libraries(mcap_recorder_test
  PRIVATE perception_recording PkgConfig::LZ4 PkgConfig::ZSTD)
target_include_directories(mcap_recorder_test PRIVATE "${CMAKE_SOURCE_DIR}/third_party")
add_test(NAME mcap_recorder_test COMMAND mcap_recorder_test)

# The shipped calibration is part of what this covers, so the test is told
# where app/config/ is rather than guessing from the working directory.
add_executable(stereo_calibration_test stereo_calibration_test.cpp)
target_link_libraries(stereo_calibration_test PRIVATE perception_camera)
target_compile_definitions(stereo_calibration_test
  PRIVATE PERCEPTION_APP_CONFIG_DIR="${CMAKE_SOURCE_DIR}/app/config")
add_test(NAME stereo_calibration_test COMMAND stereo_calibration_test)

add_executable(rectify_map_test rectify_map_test.cpp)
target_link_libraries(rectify_map_test PRIVATE perception_utils)
add_test(NAME rectify_map_test COMMAND rectify_map_test)

add_executable(cpu_debayer_test cpu_debayer_test.cpp)
target_link_libraries(cpu_debayer_test PRIVATE perception_stereo)
add_test(NAME cpu_debayer_test COMMAND cpu_debayer_test)

# --- needs a GPU, but no camera -----------------------------------------------
if(PERCEPTION_WITH_CUDA)
  add_executable(ring_pair_consumer_test ring_pair_consumer_test.cpp)
  target_link_libraries(ring_pair_consumer_test PRIVATE perception_core)
  add_test(NAME ring_pair_consumer_test COMMAND ring_pair_consumer_test)

  add_executable(upload_stage_test upload_stage_test.cpp)
  target_link_libraries(upload_stage_test PRIVATE perception_core)
  add_test(NAME upload_stage_test COMMAND upload_stage_test)

  # No GPU needed -- the ring is plain host memory -- but it lives in
  # perception_core, which links CUDA.
  add_executable(host_frame_ring_test host_frame_ring_test.cpp)
  target_link_libraries(host_frame_ring_test PRIVATE perception_core)
  add_test(NAME host_frame_ring_test COMMAND host_frame_ring_test)

  # Plain host memory too -- the holder never touches a device -- but it links
  # perception_core, which links CUDA.
  add_executable(latest_frame_test latest_frame_test.cpp)
  target_link_libraries(latest_frame_test PRIVATE perception_core)
  add_test(NAME latest_frame_test COMMAND latest_frame_test)

  # Reaches into app/include for the sink: it is the seam between the camera
  # half of the tree and this one, and it belongs to neither library.
  add_executable(ring_frame_sink_test ring_frame_sink_test.cpp)
  target_link_libraries(ring_frame_sink_test PRIVATE perception_core perception_camera)
  target_include_directories(ring_frame_sink_test PRIVATE "${CMAKE_SOURCE_DIR}/app/include")
  add_test(NAME ring_frame_sink_test COMMAND ring_frame_sink_test)

  add_executable(download_stage_test download_stage_test.cpp)
  target_link_libraries(download_stage_test PRIVATE perception_core)
  add_test(NAME download_stage_test COMMAND download_stage_test)

endif()
