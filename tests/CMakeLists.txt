set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
set(KALEIDOSCOPE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../src/kaleidoscope.c)
set(KALEIDOSCOPE_TEST_LIBS check)

# Includes
include_directories(${CMAKE_CURRENT_BINARY_DIR}/thirdparty/check/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(UNIX)
    set(KALEIDOSCOPE_TEST_LIBS check pthread)

    if(KALEIDOSCOPE_ENABLE_COVERAGE)
        include(CodeCoverage)

        add_compile_options(--coverage)
        set(KALEIDOSCOPE_TEST_LIBS check pthread gcov)
        set(GCOVR_ADDITIONAL_ARGS "--exclude-throw-branches")
        setup_target_for_coverage_gcovr_html(
            NAME
            coverage
            EXECUTABLE
            ctest -j
            DEPENDENCIES
            imageDataTest interpolationTest processingTest rotationTest slicingTest
            EXCLUDE
            "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/*"
            "${CMAKE_CURRENT_SOURCE_DIR}/../tests/*"
            )

        setup_target_for_coverage_gcovr_xml(
            NAME
            coverage-xml
            EXECUTABLE
            ctest -j
            DEPENDENCIES
            imageDataTest interpolationTest processingTest rotationTest slicingTest
            EXCLUDE
            "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/*"
            "${CMAKE_CURRENT_SOURCE_DIR}/../tests/*"
            )
    endif()
endif()

# Compile
add_executable(imageDataTest ${KALEIDOSCOPE_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/imageDataTest.c)
add_executable(interpolationTest ${KALEIDOSCOPE_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/interpolationTest.c)
add_executable(processingTest ${KALEIDOSCOPE_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/processingTest.c)
add_executable(rotationTest ${KALEIDOSCOPE_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/rotationTest.c)
add_executable(slicingTest ${KALEIDOSCOPE_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/slicingTest.c)

# Linker
target_link_libraries(imageDataTest ${KALEIDOSCOPE_TEST_LIBS})
target_link_libraries(interpolationTest ${KALEIDOSCOPE_TEST_LIBS})
target_link_libraries(processingTest ${KALEIDOSCOPE_TEST_LIBS})
target_link_libraries(rotationTest ${KALEIDOSCOPE_TEST_LIBS})
target_link_libraries(slicingTest ${KALEIDOSCOPE_TEST_LIBS})

# Enum tests
add_test(NAME imageDataTest COMMAND imageDataTest)
add_test(NAME interpolationTest COMMAND interpolationTest)
add_test(NAME processingTest COMMAND processingTest)
add_test(NAME rotationTest COMMAND rotationTest)
add_test(NAME slicingTest COMMAND slicingTest)