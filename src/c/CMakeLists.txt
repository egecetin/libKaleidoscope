add_library(kaleidoscope-static-c STATIC kaleidoscope.c)
set_target_properties(
  kaleidoscope-static-c
  PROPERTIES OUTPUT_NAME "libkaleidoscope-c" C_STANDARD 90
  PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/c/kaleidoscope.h
  PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/c/kaleidoscope-definitions.h
  VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
  SOVERSION ${PROJECT_VERSION_MAJOR})
install(TARGETS kaleidoscope-static-c)

if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
  target_link_libraries(kaleidoscope-static-c m)
endif()

if(KALEIDOSCOPE_ENABLE_SHARED)
  add_library(kaleidoscope-shared-c SHARED kaleidoscope.c)
  set_target_properties(
    kaleidoscope-shared-c
    PROPERTIES OUTPUT_NAME "libkaleidoscope-c" C_STANDARD 90
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/c/kaleidoscope.h
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/c/kaleidoscope-definitions.h
    VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
    SOVERSION ${PROJECT_VERSION_MAJOR})
  install(TARGETS kaleidoscope-shared-c)

  if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    target_link_libraries(kaleidoscope-shared-c m)
  endif()
endif()