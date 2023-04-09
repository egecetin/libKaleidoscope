cmake_minimum_required(VERSION 3.18)
project(kaleidoscope VERSION 2.0.0)

# Options
set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

option(KALEIDOSCOPE_ENABLE_SHARED "Enables shared libraries" ON)
option(KALEIDOSCOPE_ENABLE_DOCS "Enables doxygen documentation" OFF)
option(KALEIDOSCOPE_ENABLE_TESTS "Enables unit testing" OFF)
option(KALEIDOSCOPE_ENABLE_CMD_TOOL "Enables command line tool" ON)
option(KALEIDOSCOPE_ENABLE_COVERAGE "Enables coverage report" OFF)
option(KALEIDOSCOPE_ENABLE_PACKAGE "Enables packaging for library" OFF)
option(KALEIDOSCOPE_ENABLE_CUDA "Enables CUDA backend" OFF)

if(KALEIDOSCOPE_ENABLE_CUDA)
  find_package(CUDAToolkit)
endif()

if(KALEIDOSCOPE_ENABLE_CMD_TOOL)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo)
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)

if(KALEIDOSCOPE_ENABLE_TESTS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/check)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

if(KALEIDOSCOPE_ENABLE_DOCS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/doc)
endif()

if(KALEIDOSCOPE_ENABLE_PACKAGE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dist)
endif()

configure_file(kaleidoscope-config.h.in "kaleidoscope-config.h")
