cmake_minimum_required(VERSION 3.12)
project(kaleidoscope VERSION 1.3.1)

# Options
set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
set(KALEIDOSCOPE_TARGET_CUDA_ARCH "50;52;61;62;70;72;75;80;86;87;89;90")

option(KALEIDOSCOPE_ENABLE_SHARED "Enables shared libraries" ON)
option(KALEIDOSCOPE_ENABLE_DOCS "Enables doxygen documentation" OFF)
option(KALEIDOSCOPE_ENABLE_TESTS "Enables unit testing" OFF)
option(KALEIDOSCOPE_ENABLE_CMD_TOOL "Enables command line tool" ON)
option(KALEIDOSCOPE_ENABLE_COVERAGE "Enables coverage report" OFF)
option(KALEIDOSCOPE_ENABLE_PACKAGE "Enables packaging for library" OFF)
option(KALEIDOSCOPE_ENABLE_CUDA "Enables NVIDIA CUDA backend for library" OFF)
option(KALEIDOSCOPE_BUILD_PYTHON_BINDINGS "Enables to build of Python bindings" ON)

configure_file(kaleidoscope-config.h.in "kaleidoscope-config.h")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

file(GLOB KALEIDOSCOPE_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope.hpp)

# Targets
add_library(kaleidoscope-static STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope.c)
set_target_properties(
  kaleidoscope-static
  PROPERTIES OUTPUT_NAME "kaleidoscope" C_STANDARD 90
  PUBLIC_HEADER "${KALEIDOSCOPE_HEADERS}"
  VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})
install(TARGETS kaleidoscope-static)

if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
  target_link_libraries(kaleidoscope-static m)
endif()

if(KALEIDOSCOPE_ENABLE_SHARED)
  add_library(kaleidoscope-shared SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope.c)
  set_target_properties(
    kaleidoscope-shared
    PROPERTIES OUTPUT_NAME "kaleidoscope" C_STANDARD 90
    PUBLIC_HEADER "${KALEIDOSCOPE_HEADERS}"
    VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
    SOVERSION ${PROJECT_VERSION_MAJOR})
  install(TARGETS kaleidoscope-shared)

  if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    target_link_libraries(kaleidoscope-shared m)
  endif()

  if(KALEIDOSCOPE_ENABLE_CUDA)
    enable_language(CUDA)

    add_library(kaleidoscope-cuda-shared SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/kaleidoscope.cu)
    set_target_properties(
      kaleidoscope-cuda-shared
      PROPERTIES OUTPUT_NAME "kaleidoscope-cuda" CUDA_ARCHITECTURES "${KALEIDOSCOPE_TARGET_CUDA_ARCH}"
      PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/kaleidoscope.cuh
      VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
      SOVERSION ${PROJECT_VERSION_MAJOR})
    target_link_libraries(kaleidoscope-cuda-shared kaleidoscope-shared)

    install(TARGETS kaleidoscope-cuda-shared)
  endif()
elseif(KALEIDOSCOPE_ENABLE_CUDA)
  message(WARNING "CUDA backend requires shared libraries. Please enable both KALEIDOSCOPE_ENABLE_SHARED and KALEIDOSCOPE_ENABLE_CUDA")
  set(KALEIDOSCOPE_ENABLE_CUDA OFF)
endif()

if(KALEIDOSCOPE_ENABLE_CMD_TOOL)
  add_subdirectory(thirdparty/libjpeg-turbo EXCLUDE_FROM_ALL)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo)

  add_executable(kaleidoscope-cmd
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope-cmd.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/jpeg-utils/jpeg-utils.c)
  target_link_libraries(kaleidoscope-cmd kaleidoscope-static turbojpeg-static)
  set_target_properties(
    kaleidoscope-cmd
    PROPERTIES OUTPUT_NAME "kaleidoscope")
  install(TARGETS kaleidoscope-cmd)

  if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    target_link_libraries(kaleidoscope-cmd m)
  endif()
endif()

if(KALEIDOSCOPE_BUILD_PYTHON_BINDINGS)
  find_package(Python REQUIRED)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py.in "${CMAKE_CURRENT_BINARY_DIR}/python/setup.py")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/kaleidoscope.pxd ${CMAKE_CURRENT_BINARY_DIR}/python COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/kaleidoscope.pyx ${CMAKE_CURRENT_BINARY_DIR}/python COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/MANIFEST.in ${CMAKE_CURRENT_BINARY_DIR}/python COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/pyproject.toml ${CMAKE_CURRENT_BINARY_DIR}/python COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/python-tests.py ${CMAKE_CURRENT_BINARY_DIR}/python COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/README.md ${CMAKE_CURRENT_BINARY_DIR}/python COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope.hpp ${CMAKE_CURRENT_BINARY_DIR}/python COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope.h ${CMAKE_CURRENT_BINARY_DIR}/python COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope.c ${CMAKE_CURRENT_BINARY_DIR}/python COPYONLY)
  configure_file(${CMAKE_CURRENT_BINARY_DIR}/kaleidoscope-config.h ${CMAKE_CURRENT_BINARY_DIR}/python COPYONLY)

  add_custom_target(
    python-bindings ALL
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Python wheel"
    COMMAND python3 -m build ${CMAKE_CURRENT_BINARY_DIR}/python/ --sdist
  )
endif()

if(KALEIDOSCOPE_ENABLE_DOCS)
  include(Doxy)
endif()

if(KALEIDOSCOPE_ENABLE_TESTS)
  include(CTest)
  enable_testing()

  add_subdirectory(tests)
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
