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

add_executable(recording_test recording_test.cpp)
target_link_libraries(recording_test PRIVATE perception_recording)
add_test(NAME recording_test COMMAND recording_test)

add_executable(cpu_debayer_test cpu_debayer_test.cpp)
target_link_libraries(cpu_debayer_test PRIVATE perception_stereo)
add_test(NAME cpu_debayer_test COMMAND cpu_debayer_test)

# --- needs a GPU, but no camera -----------------------------------------------
if(PERCEPTION_WITH_CUDA)
  add_executable(stereo_consumer_test stereo_consumer_test.cpp)
  target_link_libraries(stereo_consumer_test PRIVATE perception_core)
  add_test(NAME stereo_consumer_test COMMAND stereo_consumer_test)

  add_executable(upload_stage_test upload_stage_test.cpp)
  target_link_libraries(upload_stage_test PRIVATE perception_core)
  add_test(NAME upload_stage_test COMMAND upload_stage_test)
endif()
