cmake_minimum_required(VERSION 3.12)
project(kaleidoscope VERSION 1.1.0)

FILE(GLOB KaleidoscopeSources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c ${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.c)

# Thirdparty libraries
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo")

# Project files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Compile
add_executable(kaleidoscope ${KaleidoscopeSources})
target_link_libraries(kaleidoscope turbojpeg-static)
