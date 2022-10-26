# Includes
include_directories(${CMAKE_CURRENT_BINARY_DIR}/thirdparty/check/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

set(KALEIDOSCOPE_TEST_LIBS check kaleidoscope-static)

# Compile
add_executable(imageDataTest imageDataTest.c)

# Linker
target_link_libraries(imageDataTest ${KALEIDOSCOPE_TEST_LIBS})

# Enum tests
add_test(NAME imageDataTest COMMAND imageDataTest)
