cmake_minimum_required(VERSION 3.12)
project(kaleidoscope VERSION 1.0.0)

# Source files
file(GLOB KaleidoscopeSources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/jpeg-utils/jpeg-utils.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope.c)

# Options
set(CMAKE_DEBUG_POSTFIX d)

option(ENABLE_SHARED "Enables shared libraries" OFF)
option(ENABLE_TESTS "Enables unit testing" OFF)
option(ENABLE_CMD_TOOL "Enables command line tool" ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Targets
add_library(kaleidoscope-static STATIC ${KaleidoscopeSources})
set_target_properties(
    kaleidoscope-static
    PROPERTIES OUTPUT_NAME "kaleidoscope" C_STANDARD 90
    VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
    SOVERSION ${PROJECT_VERSION_MAJOR})

if(CMAKE_C_COMPILER_ID MATCHES "GNU"
     OR CMAKE_CXX_COMPILER_ID MATCHES "GNU"
     OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang"
     OR CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    target_link_libraries(kaleidoscope-static m)
endif()

if(ENABLE_SHARED)
    add_library(kaleidoscope-shared SHARED ${KaleidoscopeSources})
    set_target_properties(
        kaleidoscope-shared
        PROPERTIES OUTPUT_NAME "kaleidoscope" C_STANDARD 90
        VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
        SOVERSION ${PROJECT_VERSION_MAJOR})

    if(CMAKE_C_COMPILER_ID MATCHES "GNU"
        OR CMAKE_CXX_COMPILER_ID MATCHES "GNU"
        OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang"
        OR CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
        target_link_libraries(kaleidoscope-static m)
    endif()
endif()

if(ENABLE_CMD_TOOL)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libjpeg-turbo)

    add_executable(kaleidoscope-cmd ${CMAKE_CURRENT_SOURCE_DIR}/src/kaleidoscope-cmd.c)
    target_link_libraries(kaleidoscope-cmd kaleidoscope-static turbojpeg-static)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU"
        OR CMAKE_CXX_COMPILER_ID MATCHES "GNU"
        OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang"
        OR CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
        target_link_libraries(kaleidoscope-cmd m)
    endif()
endif()

if(ENABLE_TESTS)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()
