cmake_minimum_required(VERSION 3.12)
project(kaleidoscope VERSION 2.0.0)

# Options
set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

option(KALEIDOSCOPE_ENABLE_SHARED "Enables shared libraries" OFF)
option(KALEIDOSCOPE_ENABLE_DOCS "Enables doxygen documentation" OFF)
option(KALEIDOSCOPE_ENABLE_TESTS "Enables unit testing" OFF)
option(KALEIDOSCOPE_ENABLE_CMD_TOOL "Enables command line tool" ON)
option(KALEIDOSCOPE_ENABLE_COVERAGE "Enables coverage report" OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Targets
add_library(kaleidoscope-static STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope.c)
set_target_properties(
  kaleidoscope-static
  PROPERTIES OUTPUT_NAME "kaleidoscope" C_STANDARD 90
  VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
  SOVERSION ${PROJECT_VERSION_MAJOR})

if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
  target_link_libraries(kaleidoscope-static m)
endif()

if(KALEIDOSCOPE_ENABLE_SHARED)
  add_library(kaleidoscope-shared SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope.c)
  set_target_properties(
    kaleidoscope-shared
    PROPERTIES OUTPUT_NAME "kaleidoscope" C_STANDARD 90
    VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
    SOVERSION ${PROJECT_VERSION_MAJOR})

  if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    target_link_libraries(kaleidoscope-static m)
  endif()
endif()

if(KALEIDOSCOPE_ENABLE_CMD_TOOL)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo)

  add_executable(kaleidoscope-cmd
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope-cmd.c
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/jpeg-utils/jpeg-utils.c)
  target_link_libraries(kaleidoscope-cmd kaleidoscope-static turbojpeg-static)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    target_link_libraries(kaleidoscope-cmd m)
  endif()
endif()

if(KALEIDOSCOPE_ENABLE_DOCS)
  include(Doxy)
endif()

if(KALEIDOSCOPE_ENABLE_TESTS)
  include(CTest)
  enable_testing()

  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/check")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()
