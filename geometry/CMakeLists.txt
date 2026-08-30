find_package(Eigen3 REQUIRED)

add_library(perception_geometry
  src/camera_model.cpp
  src/stereo.cpp)

target_include_directories(perception_geometry PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(perception_geometry PUBLIC Eigen3::Eigen)

if(PERCEPTION_BUILD_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    add_subdirectory(cpp_test)
  else()
    message(STATUS "GTest not found -- skipping perception_geometry tests")
  endif()
endif()
