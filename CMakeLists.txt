cmake_minimum_required(VERSION 3.12)

project(kaleidoscope VERSION 1.1.0)

# Compile options
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -DNDEBUG")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin")

FILE(GLOB ProjectSources ${PROJECT_SOURCE_DIR}/src/*.c ${PROJECT_SOURCE_DIR}/src/**/*.c)

# Thirdparty libraries
add_subdirectory("thirdparty/libjpeg-turbo")

# Project files
include_directories(thirdparty/libjpeg-turbo)
include_directories(${PROJECT_SOURCE_DIR}/include)

# Compile
add_executable(kaleidoscope ${ProjectSources})
target_link_libraries(kaleidoscope turbojpeg-static)
