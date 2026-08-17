# Behaviour tests for pipeline code, as opposed to smoke/, which only checks
# that the toolchain works. These need a GPU but no camera.

add_executable(upload_stage_test upload_stage_test.cpp)
target_link_libraries(upload_stage_test PRIVATE perception_core)
add_test(NAME upload_stage_test COMMAND upload_stage_test)
