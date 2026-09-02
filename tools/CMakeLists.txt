add_subdirectory(spin_acquire)
add_subdirectory(stereo_view)
add_subdirectory(ptp_trigger_test)

if(PERCEPTION_WITH_ROS)
  add_subdirectory(ros_schemas)
endif()
