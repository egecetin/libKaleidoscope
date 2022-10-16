cmake_minimum_required(VERSION 3.12)
project(kaleidoscope VERSION 2.0.0)

file(GLOB KaleidoscopeSources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/jpeg-utils/jpeg-utils.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope.c)

# Thirdparty libraries
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo")

# Project files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Compile
add_library(kaleidoscope-static ${KaleidoscopeSources})

add_executable(kaleidoscope-cmd ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope-cmd.c)
target_link_libraries(kaleidoscope-cmd kaleidoscope-static turbojpeg-static)
