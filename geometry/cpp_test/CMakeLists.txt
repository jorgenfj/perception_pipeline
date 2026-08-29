include(GoogleTest)

set(TEST_BINARY_NAME perception_geometry_test)

add_executable(${TEST_BINARY_NAME}
  test_camera_model.cpp)

target_link_libraries(${TEST_BINARY_NAME}
  PRIVATE
  perception_geometry
  GTest::gtest
  GTest::gtest_main)

gtest_discover_tests(${TEST_BINARY_NAME})
