enable_language(CUDA)
add_library(kaleidoscope-cuda-static STATIC kaleidoscope.cu)
target_link_libraries(kaleidoscope-cuda-static kaleidoscope-static-cpp)
set_target_properties(
    kaleidoscope-cuda-static
    PROPERTIES OUTPUT_NAME "libkaleidoscope-cuda" CUDA_ARCHITECTURES OFF
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope.h
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope-definitions.h
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/kaleidoscope-cuda.hpp
    VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
    SOVERSION ${PROJECT_VERSION_MAJOR})
install(TARGETS kaleidoscope-cuda-static)

if(KALEIDOSCOPE_ENABLE_SHARED)
    add_library(kaleidoscope-cuda-shared SHARED kaleidoscope.cu)
    target_link_libraries(kaleidoscope-cuda-shared kaleidoscope-shared-cpp)
    set_target_properties(
        kaleidoscope-cuda-shared
        PROPERTIES OUTPUT_NAME "libkaleidoscope-cuda" CUDA_ARCHITECTURES OFF
        PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope.h
        PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/kaleidoscope-definitions.h
        PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/kaleidoscope-cuda.hpp
        VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
        SOVERSION ${PROJECT_VERSION_MAJOR})
    install(TARGETS kaleidoscope-cuda-shared)
endif()