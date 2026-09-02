if(NOT ROS_SHARE_DIR AND DEFINED ENV{ROS_DISTRO})
  set(ROS_SHARE_DIR "/opt/ros/$ENV{ROS_DISTRO}/share" CACHE PATH "ROS 2 share/ directory")
endif()
if(NOT ROS_SHARE_DIR OR NOT IS_DIRECTORY "${ROS_SHARE_DIR}")
  message(FATAL_ERROR
    "PERCEPTION_WITH_ROS is on but no ROS message tree was found. Source a ROS 2 "
    "setup.bash, or pass -DROS_SHARE_DIR=/path/to/share. Looked at: '${ROS_SHARE_DIR}'")
endif()
message(STATUS "ROS messages: ${ROS_SHARE_DIR}")

add_executable(ros_schema_gen ros_schema_gen.cpp)

set(_generated "${CMAKE_SOURCE_DIR}/recording/include/ros_schemas.hpp")
set(_args
  --search "${ROS_SHARE_DIR}"
  --search "${CMAKE_SOURCE_DIR}/recording/share"
  --types "${CMAKE_CURRENT_SOURCE_DIR}/types.txt")

add_custom_target(ros_schemas
  COMMAND ros_schema_gen ${_args} --out "${_generated}"
  COMMENT "Generating recording/include/ros_schemas.hpp"
  VERBATIM)

add_test(NAME ros_schemas_check COMMAND ros_schema_gen ${_args} --out "${_generated}" --check)
