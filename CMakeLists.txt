cmake_minimum_required(VERSION 3.12)

project(kaleidoscope VERSION 1.0.0)

# Compile options
add_compile_options(-Wall -Wextra -pedantic)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O0 -g -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -fopenmp")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -DNDEBUG -O3 -fopenmp -flto")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -fopenmp -flto")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin")

FILE(GLOB ProjectSources ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.c ${PROJECT_SOURCE_DIR}/src/**/*.cpp ${PROJECT_SOURCE_DIR}/src/**/*.c)

# Thirdparty libraries
add_subdirectory("thirdparty/libjpeg-turbo")

# Project files
include_directories(thirdparty/libjpeg-turbo)
include_directories(${PROJECT_SOURCE_DIR}/include)

# Compile
add_executable(kaleidoscope ${ProjectSources})
target_link_libraries(kaleidoscope turbojpeg-static m)
