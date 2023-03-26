cmake_minimum_required(VERSION 3.12)
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
  find_package(CUDA)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Targets
add_library(kaleidoscope-static STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope.c)
set_target_properties(
  kaleidoscope-static
  PROPERTIES OUTPUT_NAME "kaleidoscope" C_STANDARD 90
  PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope.h
  PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope-definitions.h
  VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
  SOVERSION ${PROJECT_VERSION_MAJOR})
install(TARGETS kaleidoscope-static)

if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
  target_link_libraries(kaleidoscope-static m)
endif()

if(KALEIDOSCOPE_ENABLE_SHARED)
  add_library(kaleidoscope-shared SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope.c)
  set_target_properties(
    kaleidoscope-shared
    PROPERTIES OUTPUT_NAME "kaleidoscope" C_STANDARD 90
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope.h
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope-definitions.h
    VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
    SOVERSION ${PROJECT_VERSION_MAJOR})
  install(TARGETS kaleidoscope-shared)

  if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    target_link_libraries(kaleidoscope-static m)
  endif()
endif()

if(NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Disabling GPU support (KALEIDOSCOPE_ENABLE_CUDA = 0)")
  set(KALEIDOSCOPE_ENABLE_CUDA 0)
else()
  message(STATUS "CUDA found enabling GPU support (KALEIDOSCOPE_ENABLE_CUDA = 1)")
  enable_language(CUDA)
  add_library(kaleidoscope-cuda STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope.cu)
  target_link_libraries(kaleidoscope-cuda kaleidoscope-static)
  set_target_properties(
    kaleidoscope-cuda
    PROPERTIES OUTPUT_NAME "kaleidoscope-cuda"
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope.h
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope-cuda.hpp
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope-definitions.h
    VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
    SOVERSION ${PROJECT_VERSION_MAJOR})
  install(TARGETS kaleidoscope-cuda)
endif()

if(KALEIDOSCOPE_ENABLE_CMD_TOOL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo EXCLUDE_FROM_ALL)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo)

  add_executable(kaleidoscope-cmd
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope-cmd.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/jpeg-utils/jpeg-utils.c)
  target_link_libraries(kaleidoscope-cmd kaleidoscope-shared turbojpeg-static)
  set_target_properties(
    kaleidoscope-cmd
    PROPERTIES OUTPUT_NAME "kaleidoscope")
  install(TARGETS kaleidoscope-cmd)

  if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    target_link_libraries(kaleidoscope-cmd m)
  endif()

  if(NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Command line tool for CUDA backend disabling")
  else()
    message(STATUS "CUDA found. Command line tool for CUDA backend enabling")
    add_executable(kaleidoscope-cmd-cuda
      ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope-cmd-cuda.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/src/jpeg-utils/jpeg-utils.c)
    target_link_libraries(kaleidoscope-cmd-cuda kaleidoscope-cuda turbojpeg-static)
    set_target_properties(
      kaleidoscope-cmd-cuda
      PROPERTIES OUTPUT_NAME "kaleidoscope-cuda")
    install(TARGETS kaleidoscope-cmd-cuda)
  endif()
endif()

if(KALEIDOSCOPE_ENABLE_DOCS)
  include(Doxy)
endif()

if(KALEIDOSCOPE_ENABLE_TESTS)
  include(CTest)
  enable_testing()

  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/check EXCLUDE_FROM_ALL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

if(KALEIDOSCOPE_ENABLE_PACKAGE)
  set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})
  set(CPACK_PACKAGE_NAME "kaleidoscope")
  set(CPACK_PACKAGE_RELEASE ${PROJECT_VERSION_PATCH})
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}.${CMAKE_SYSTEM_PROCESSOR}")
  set(CPACK_OUTPUT_FILE_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/dist)
  set(CPACK_PACKAGE_MAINTAINER "egecetin")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A library to create kaleidoscope effect on images")
  set(CPACK_RESOURCE_FILE_LICENSE ${PROJECT_SOURCE_DIR}/LICENSE)
  set(CPACK_RESOURCE_FILE_README ${PROJECT_SOURCE_DIR}/README.md)
  set(CPACK_PACKAGE_CONTACT "egecetin@hotmail.com.tr")

  if(APPLE)
    set(CPACK_GENERATOR "DragNDrop")
  elseif(UNIX)
    set(CPACK_GENERATOR "DEB")
  elseif(WIN32)
    set(CPACK_GENERATOR "NSIS")
  endif()

  include(CPack)
endif()

configure_file(kaleidoscope-config.h.in "kaleidoscope-config.h")
