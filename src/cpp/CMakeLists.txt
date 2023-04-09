add_library(kaleidoscope-static-cpp STATIC kaleidoscope.cpp)
set_target_properties(
    kaleidoscope-static-cpp
    PROPERTIES OUTPUT_NAME "libkaleidoscope-cpp" CXX_STANDARD 11
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/c/kaleidoscope.h
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/c/kaleidoscope-definitions.h
    VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
    SOVERSION ${PROJECT_VERSION_MAJOR})
install(TARGETS kaleidoscope-static-cpp)

if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    target_link_libraries(kaleidoscope-static-cpp m)
endif()
target_link_libraries(kaleidoscope-static-cpp kaleidoscope-static-c)

if(KALEIDOSCOPE_ENABLE_SHARED)
    add_library(kaleidoscope-shared-cpp SHARED kaleidoscope.cpp)
    set_target_properties(
        kaleidoscope-shared-cpp
        PROPERTIES OUTPUT_NAME "libkaleidoscope-cpp" CXX_STANDARD 11
        PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/c/kaleidoscope.h
        PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/c/kaleidoscope-definitions.h
        VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
        SOVERSION ${PROJECT_VERSION_MAJOR})
    install(TARGETS kaleidoscope-shared-cpp)

    if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
        target_link_libraries(kaleidoscope-shared-cpp m)
    endif()
    target_link_libraries(kaleidoscope-shared-cpp kaleidoscope-shared-c)
endif()

if(NOT CUDAToolkit_FOUND)
    message(STATUS "CUDA not found. Disabling GPU support (KALEIDOSCOPE_ENABLE_CUDA = 0)")
    set(KALEIDOSCOPE_ENABLE_CUDA 0)
else()
    message(STATUS "CUDA found. Enabling GPU support (KALEIDOSCOPE_ENABLE_CUDA = 1)")
    set(KALEIDOSCOPE_ENABLE_CUDA 1)
    add_subdirectory(cuda)
endif()