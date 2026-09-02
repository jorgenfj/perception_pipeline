find_package(Eigen3 REQUIRED)

add_library(perception_utils
  src/camera_model.cpp
  src/rectification.cpp
  src/stereo_validate.cpp)

target_include_directories(perception_utils PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(perception_utils PUBLIC Eigen3::Eigen)

if(PERCEPTION_BUILD_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    add_subdirectory(cpp_test)
  else()
    message(STATUS "GTest not found -- skipping perception_utils tests")
  endif()
endif()
