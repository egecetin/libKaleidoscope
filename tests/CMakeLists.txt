# Includes
include_directories(${CMAKE_CURRENT_BINARY_DIR}/thirdparty/check/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(KALEIDOSCOPE_TEST_LIBS check kaleidoscope-static)

# Compile
add_executable(imageDataTest ${CMAKE_CURRENT_SOURCE_DIR}/src/imageDataTest.c)
add_executable(interpolationTest ${CMAKE_CURRENT_SOURCE_DIR}/src/interpolationTest.c)
# add_executable(processingTest ${CMAKE_CURRENT_SOURCE_DIR}/src/processingTest.c)
add_executable(rotationTest ${CMAKE_CURRENT_SOURCE_DIR}/src/rotationTest.c)
add_executable(slicingTest ${CMAKE_CURRENT_SOURCE_DIR}/src/slicingTest.c)

# Linker
target_link_libraries(imageDataTest ${KALEIDOSCOPE_TEST_LIBS})
target_link_libraries(interpolationTest ${KALEIDOSCOPE_TEST_LIBS})
# target_link_libraries(processingTest ${KALEIDOSCOPE_TEST_LIBS})
target_link_libraries(rotationTest ${KALEIDOSCOPE_TEST_LIBS})
target_link_libraries(slicingTest ${KALEIDOSCOPE_TEST_LIBS})

# Enum tests
add_test(NAME imageDataTest COMMAND imageDataTest)
add_test(NAME interpolationTest COMMAND interpolationTest)
# add_test(NAME processingTest COMMAND processingTest)
add_test(NAME rotationTest COMMAND rotationTest)
add_test(NAME slicingTest COMMAND slicingTest)
