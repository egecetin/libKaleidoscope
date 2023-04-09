add_subdirectory(c)
add_subdirectory(cpp)

if(KALEIDOSCOPE_ENABLE_CMD_TOOL AND KALEIDOSCOPE_ENABLE_SHARED)
    add_executable(kaleidoscope-cmd
        kaleidoscope-cmd.cpp
        c/jpeg-utils/jpeg-utils.c)
    target_link_libraries(kaleidoscope-cmd kaleidoscope-shared turbojpeg-static)
    set_target_properties(
        kaleidoscope-cmd
        PROPERTIES OUTPUT_NAME "libkaleidoscope")
    install(TARGETS kaleidoscope-cmd)

    if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
        target_link_libraries(kaleidoscope-cmd m)
    endif()

    if(NOT CUDAToolkit_FOUND)
        message(STATUS "CUDA not found. Disabling command line tool for CUDA backend")
    else()
        message(STATUS "CUDA found. Enabling command line tool for CUDA backend")
        target_link_libraries(kaleidoscope-cmd kaleidoscope-cuda)
    endif()
endif()