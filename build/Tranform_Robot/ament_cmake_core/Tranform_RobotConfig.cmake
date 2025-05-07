# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_Tranform_Robot_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED Tranform_Robot_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(Tranform_Robot_FOUND FALSE)
  elseif(NOT Tranform_Robot_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(Tranform_Robot_FOUND FALSE)
  endif()
  return()
endif()
set(_Tranform_Robot_CONFIG_INCLUDED TRUE)

# output package information
if(NOT Tranform_Robot_FIND_QUIETLY)
  message(STATUS "Found Tranform_Robot: 0.0.0 (${Tranform_Robot_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'Tranform_Robot' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${Tranform_Robot_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(Tranform_Robot_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${Tranform_Robot_DIR}/${_extra}")
endforeach()
