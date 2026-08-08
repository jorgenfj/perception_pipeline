# Locates the FLIR/Teledyne Spinnaker SDK and defines Spinnaker::Spinnaker.
#
# Searches SPINNAKER_ROOT first, then the two layouts FLIR ships: /opt/spinnaker
# (tarball, and what this project's container uses) and /usr (Debian packages).

find_path(Spinnaker_INCLUDE_DIR
  NAMES Spinnaker.h
  HINTS ${SPINNAKER_ROOT} $ENV{SPINNAKER_ROOT} /opt/spinnaker /usr /usr/local
  PATH_SUFFIXES include include/spinnaker)

find_library(Spinnaker_LIBRARY
  NAMES Spinnaker
  HINTS ${SPINNAKER_ROOT} $ENV{SPINNAKER_ROOT} /opt/spinnaker /usr /usr/local
  PATH_SUFFIXES lib lib64)

# The version macros live in System.h (there is no SpinnakerVersion.h in 4.x).
if(Spinnaker_INCLUDE_DIR AND EXISTS "${Spinnaker_INCLUDE_DIR}/System.h")
  file(STRINGS "${Spinnaker_INCLUDE_DIR}/System.h" _spin_ver_lines
       REGEX "#define +FLIR_SPINNAKER_VERSION_(MAJOR|MINOR|TYPE|BUILD)")
  foreach(_part MAJOR MINOR TYPE BUILD)
    if(_spin_ver_lines MATCHES "#define +FLIR_SPINNAKER_VERSION_${_part} +([0-9]+)")
      set(_spin_${_part} "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  if(DEFINED _spin_MAJOR)
    set(Spinnaker_VERSION "${_spin_MAJOR}.${_spin_MINOR}.${_spin_TYPE}.${_spin_BUILD}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Spinnaker
  REQUIRED_VARS Spinnaker_LIBRARY Spinnaker_INCLUDE_DIR
  VERSION_VAR Spinnaker_VERSION)

if(Spinnaker_FOUND AND NOT TARGET Spinnaker::Spinnaker)
  add_library(Spinnaker::Spinnaker UNKNOWN IMPORTED)
  set_target_properties(Spinnaker::Spinnaker PROPERTIES
    IMPORTED_LOCATION "${Spinnaker_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Spinnaker_INCLUDE_DIR}")
endif()

mark_as_advanced(Spinnaker_INCLUDE_DIR Spinnaker_LIBRARY)
