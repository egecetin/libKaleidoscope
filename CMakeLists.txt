cmake_minimum_required(VERSION 3.12)
project(kaleidoscope VERSION 1.0.0)

file(GLOB KaleidoscopeSources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/jpeg-utils/jpeg-utils.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope.c)

set(CMAKE_DEBUG_POSTFIX d)

option(ENABLE_TESTS "Enables unit testing" OFF)
option(ENABLE_CMD_TOOL "Enables command line tool" ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if (ENABLE_CMD_TOOL)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo)
endif()

if (ENABLE_TESTS)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()

# Compile
add_library(kaleidoscope-static ${KaleidoscopeSources})
if(CMAKE_C_COMPILER_ID MATCHES "GNU"
     OR CMAKE_CXX_COMPILER_ID MATCHES "GNU"
     OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang"
     OR CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    target_link_libraries(kaleidoscope-static m)
endif()

add_executable(kaleidoscope-cmd ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope-cmd.c)
target_link_libraries(kaleidoscope-cmd kaleidoscope-static turbojpeg-static)
if(CMAKE_C_COMPILER_ID MATCHES "GNU"
     OR CMAKE_CXX_COMPILER_ID MATCHES "GNU"
     OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang"
     OR CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    target_link_libraries(kaleidoscope-cmd m)
endif()
