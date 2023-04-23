set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
set(KALEIDOSCOPE_TEST_LIBS check kaleidoscope-static)

# Includes
include_directories(${CMAKE_CURRENT_BINARY_DIR}/thirdparty/check/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(UNIX)
    list(APPEND KALEIDOSCOPE_TEST_LIBS pthread)

    if(KALEIDOSCOPE_ENABLE_COVERAGE)
        include(CodeCoverage)

        add_compile_options(--coverage)
        list(APPEND KALEIDOSCOPE_TEST_LIBS gcov)
        set(GCOVR_ADDITIONAL_ARGS "--exclude-throw-branches")
        setup_target_for_coverage_gcovr_html(
            NAME
            coverage
            EXECUTABLE
            ctest -j
            DEPENDENCIES
            interpolationTest processingTest rotationTest slicingTest versionTest
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
            interpolationTest processingTest rotationTest slicingTest versionTest
            EXCLUDE
            "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/*"
            "${CMAKE_CURRENT_SOURCE_DIR}/../tests/*"
            )
    endif()
endif()

# Compile
add_executable(interpolationTest ${CMAKE_CURRENT_SOURCE_DIR}/src/interpolationTest.c)
add_executable(processingTest ${CMAKE_CURRENT_SOURCE_DIR}/src/processingTest.c)
add_executable(rotationTest ${CMAKE_CURRENT_SOURCE_DIR}/src/rotationTest.c)
add_executable(slicingTest ${CMAKE_CURRENT_SOURCE_DIR}/src/slicingTest.c)
add_executable(versionTest ${CMAKE_CURRENT_SOURCE_DIR}/src/versionTest.c)

# Linker
target_link_libraries(interpolationTest ${KALEIDOSCOPE_TEST_LIBS})
target_link_libraries(processingTest ${KALEIDOSCOPE_TEST_LIBS})
target_link_libraries(rotationTest ${KALEIDOSCOPE_TEST_LIBS})
target_link_libraries(slicingTest ${KALEIDOSCOPE_TEST_LIBS})
target_link_libraries(versionTest ${KALEIDOSCOPE_TEST_LIBS})

# Enum tests
add_test(NAME interpolationTest COMMAND interpolationTest)
add_test(NAME processingTest COMMAND processingTest)
add_test(NAME rotationTest COMMAND rotationTest)
add_test(NAME slicingTest COMMAND slicingTest)
add_test(NAME versionTest COMMAND versionTest)
