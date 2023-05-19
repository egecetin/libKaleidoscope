set(gtest_force_shared_crt on)

add_subdirectory(check EXCLUDE_FROM_ALL)
add_subdirectory(gtest EXCLUDE_FROM_ALL)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

# Includes
include_directories(check/src)
include_directories(gtest/googletest/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Compile
set(KALEIDOSCOPE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../src/kaleidoscope.c)
add_executable(interpolationTest ${KALEIDOSCOPE_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/interpolationTest.c)
add_executable(processingTest ${KALEIDOSCOPE_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/processingTest.c)
add_executable(rotationTest ${KALEIDOSCOPE_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/rotationTest.c)
add_executable(slicingTest ${KALEIDOSCOPE_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/slicingTest.c)
add_executable(versionTest ${KALEIDOSCOPE_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/versionTest.c)

add_executable(processingTestCpp ${CMAKE_CURRENT_SOURCE_DIR}/src/processingTest.cpp)

list(APPEND KALEIDOSCOPE_C_TEST_LIBS check)
list(APPEND KALEIDOSCOPE_CPP_TEST_LIBS gtest_main)
list(APPEND KALEIDOSCOPE_TEST_TARGETS interpolationTest processingTest rotationTest slicingTest versionTest processingTestCpp)

if(KALEIDOSCOPE_ENABLE_CUDA)
    add_executable(processingTestCuda ${CMAKE_CURRENT_SOURCE_DIR}/src/processingTest.cu)
    set_target_properties(processingTestCuda PROPERTIES CUDA_ARCHITECTURES "${KALEIDOSCOPE_TARGET_CUDA_ARCH}")
    list(APPEND KALEIDOSCOPE_TEST_TARGETS processingTestCuda)
endif()

if(UNIX)
    list(APPEND KALEIDOSCOPE_C_TEST_LIBS pthread)

    if(KALEIDOSCOPE_ENABLE_COVERAGE)
        include(CodeCoverage)

        add_compile_options(--coverage)
        add_link_options(--coverage)
        set(GCOVR_ADDITIONAL_ARGS "--exclude-throw-branches")
        setup_target_for_coverage_gcovr_html(
            NAME
            coverage
            EXECUTABLE
            ctest -j
            DEPENDENCIES
            "${KALEIDOSCOPE_TEST_TARGETS}"
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
            "${KALEIDOSCOPE_TEST_TARGETS}"
            EXCLUDE
            "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/*"
            "${CMAKE_CURRENT_SOURCE_DIR}/../tests/*"
        )
    endif()
endif()

# Linker
target_link_libraries(interpolationTest ${KALEIDOSCOPE_C_TEST_LIBS})
target_link_libraries(processingTest ${KALEIDOSCOPE_C_TEST_LIBS})
target_link_libraries(rotationTest ${KALEIDOSCOPE_C_TEST_LIBS})
target_link_libraries(slicingTest ${KALEIDOSCOPE_C_TEST_LIBS})
target_link_libraries(versionTest ${KALEIDOSCOPE_C_TEST_LIBS})

target_link_libraries(processingTestCpp ${KALEIDOSCOPE_CPP_TEST_LIBS} kaleidoscope-static)

if(KALEIDOSCOPE_ENABLE_CUDA)
    target_link_libraries(processingTestCuda ${KALEIDOSCOPE_CPP_TEST_LIBS} kaleidoscope-cuda-shared)
endif()

# Enum tests
add_test(NAME interpolationTest COMMAND interpolationTest)
add_test(NAME processingTest COMMAND processingTest)
add_test(NAME rotationTest COMMAND rotationTest)
add_test(NAME slicingTest COMMAND slicingTest)
add_test(NAME versionTest COMMAND versionTest)
add_test(NAME processingTestCpp COMMAND processingTestCpp)

if(KALEIDOSCOPE_ENABLE_CUDA)
    add_test(NAME processingTestCuda COMMAND processingTestCuda)
endif()
