include(GoogleTest)

set(TEST_BINARY_NAME perception_utils_test)

add_executable(${TEST_BINARY_NAME}
  test_camera_model.cpp)

target_link_libraries(${TEST_BINARY_NAME}
  PRIVATE
  perception_utils
  GTest::gtest
  GTest::gtest_main)

gtest_discover_tests(${TEST_BINARY_NAME})
