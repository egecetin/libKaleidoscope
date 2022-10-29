# Copyright (c) 2012 - 2017, Lars Bilke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

include(CMakeParseArguments)

option(CODE_COVERAGE_VERBOSE "Verbose information" FALSE)

# Check prereqs
find_program(GCOV_PATH gcov)
find_program(GENHTML_PATH NAMES genhtml genhtml.perl genhtml.bat)
find_program(GCOVR_PATH gcovr PATHS ${CMAKE_SOURCE_DIR}/scripts/test)
find_program(CPPFILT_PATH NAMES c++filt)

if(NOT GCOV_PATH)
  message(FATAL_ERROR "gcov not found! Aborting...")
endif() # NOT GCOV_PATH

get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
list(GET LANGUAGES 0 LANG)

if("${CMAKE_${LANG}_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang")
  if("${CMAKE_${LANG}_COMPILER_VERSION}" VERSION_LESS 3)
    message(FATAL_ERROR "Clang version must be 3.0.0 or greater! Aborting...")
  endif()
elseif(NOT CMAKE_COMPILER_IS_GNUCXX)
  if("${CMAKE_Fortran_COMPILER_ID}" MATCHES "[Ff]lang")
    # Do nothing; exit conditional without error if true
  elseif("${CMAKE_Fortran_COMPILER_ID}" MATCHES "GNU")
    # Do nothing; exit conditional without error if true
  else()
    message(FATAL_ERROR "Compiler is not GNU gcc! Aborting...")
  endif()
endif()

set(COVERAGE_COMPILER_FLAGS
    "-g -fprofile-arcs -ftest-coverage"
    CACHE INTERNAL "")
if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-fprofile-abs-path HAVE_fprofile_abs_path)
  if(HAVE_fprofile_abs_path)
    set(COVERAGE_COMPILER_FLAGS "${COVERAGE_COMPILER_FLAGS} -fprofile-abs-path")
  endif()
endif()

set(CMAKE_Fortran_FLAGS_COVERAGE
    ${COVERAGE_COMPILER_FLAGS}
    CACHE STRING "Flags used by the Fortran compiler during coverage builds." FORCE)
set(CMAKE_CXX_FLAGS_COVERAGE
    ${COVERAGE_COMPILER_FLAGS}
    CACHE STRING "Flags used by the C++ compiler during coverage builds." FORCE)
set(CMAKE_C_FLAGS_COVERAGE
    ${COVERAGE_COMPILER_FLAGS}
    CACHE STRING "Flags used by the C compiler during coverage builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE
    ""
    CACHE STRING "Flags used for linking binaries during coverage builds." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE
    ""
    CACHE STRING "Flags used by the shared libraries linker during coverage builds." FORCE)
mark_as_advanced(CMAKE_Fortran_FLAGS_COVERAGE CMAKE_CXX_FLAGS_COVERAGE CMAKE_C_FLAGS_COVERAGE
                 CMAKE_EXE_LINKER_FLAGS_COVERAGE CMAKE_SHARED_LINKER_FLAGS_COVERAGE)

get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT (CMAKE_BUILD_TYPE STREQUAL "Debug" OR GENERATOR_IS_MULTI_CONFIG))
  message(WARNING "Code coverage results with an optimised (non-Debug) build may be misleading")
endif() # NOT (CMAKE_BUILD_TYPE STREQUAL "Debug" OR GENERATOR_IS_MULTI_CONFIG)

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  link_libraries(gcov)
endif()

# Defines a target for running and collection code coverage information
# Builds dependencies, runs the given executable and outputs reports.
# NOTE! The executable should always have a ZERO as exit code otherwise
# the coverage generation will not complete.
#
# setup_target_for_coverage_gcovr_xml(
#     NAME ctest_coverage                    # New target name
#     EXECUTABLE ctest -j ${PROCESSOR_COUNT} # Executable in PROJECT_BINARY_DIR
#     DEPENDENCIES executable_target         # Dependencies to build first
#     BASE_DIRECTORY "../"                   # Base directory for report
#                                            #  (defaults to PROJECT_SOURCE_DIR)
#     EXCLUDE "src/dir1/*" "src/dir2/*"      # Patterns to exclude (can be relative
#                                            #  to BASE_DIRECTORY, with CMake 3.4+)
# )
# The user can set the variable GCOVR_ADDITIONAL_ARGS to supply additional flags to the
# GCVOR command.
function(setup_target_for_coverage_gcovr_xml)

  set(options NONE)
  set(oneValueArgs BASE_DIRECTORY NAME)
  set(multiValueArgs EXCLUDE EXECUTABLE EXECUTABLE_ARGS DEPENDENCIES)
  cmake_parse_arguments(Coverage "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT GCOVR_PATH)
    message(FATAL_ERROR "gcovr not found! Aborting...")
  endif() # NOT GCOVR_PATH

  # Set base directory (as absolute path), or default to PROJECT_SOURCE_DIR
  if(DEFINED Coverage_BASE_DIRECTORY)
    get_filename_component(BASEDIR ${Coverage_BASE_DIRECTORY} ABSOLUTE)
  else()
    set(BASEDIR ${PROJECT_SOURCE_DIR})
  endif()

  # Collect excludes (CMake 3.4+: Also compute absolute paths)
  set(GCOVR_EXCLUDES "")
  foreach(EXCLUDE ${Coverage_EXCLUDE} ${COVERAGE_EXCLUDES} ${COVERAGE_GCOVR_EXCLUDES})
    if(CMAKE_VERSION VERSION_GREATER 3.4)
      get_filename_component(EXCLUDE ${EXCLUDE} ABSOLUTE BASE_DIR ${BASEDIR})
    endif()
    list(APPEND GCOVR_EXCLUDES "${EXCLUDE}")
  endforeach()
  list(REMOVE_DUPLICATES GCOVR_EXCLUDES)

  # Combine excludes to several -e arguments
  set(GCOVR_EXCLUDE_ARGS "")
  foreach(EXCLUDE ${GCOVR_EXCLUDES})
    list(APPEND GCOVR_EXCLUDE_ARGS "-e")
    list(APPEND GCOVR_EXCLUDE_ARGS "${EXCLUDE}")
  endforeach()

  # Set up commands which will be run to generate coverage data Run tests
  set(GCOVR_XML_EXEC_TESTS_CMD ${Coverage_EXECUTABLE} ${Coverage_EXECUTABLE_ARGS})
  # Running gcovr
  set(GCOVR_XML_CMD
      ${GCOVR_PATH}
      --xml
      ${Coverage_NAME}.xml
      -r
      ${BASEDIR}
      ${GCOVR_ADDITIONAL_ARGS}
      ${GCOVR_EXCLUDE_ARGS}
      --object-directory=${PROJECT_BINARY_DIR})

  if(CODE_COVERAGE_VERBOSE)
    message(STATUS "Executed command report")

    message(STATUS "Command to run tests: ")
    string(REPLACE ";" " " GCOVR_XML_EXEC_TESTS_CMD_SPACED "${GCOVR_XML_EXEC_TESTS_CMD}")
    message(STATUS "${GCOVR_XML_EXEC_TESTS_CMD_SPACED}")

    message(STATUS "Command to generate gcovr XML coverage data: ")
    string(REPLACE ";" " " GCOVR_XML_CMD_SPACED "${GCOVR_XML_CMD}")
    message(STATUS "${GCOVR_XML_CMD_SPACED}")
  endif()

  add_custom_target(
    ${Coverage_NAME}
    COMMAND ${GCOVR_XML_EXEC_TESTS_CMD}
    COMMAND ${GCOVR_XML_CMD}
    BYPRODUCTS ${Coverage_NAME}.xml
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    DEPENDS ${Coverage_DEPENDENCIES}
    VERBATIM # Protect arguments to commands
    COMMENT "Running gcovr to produce Cobertura code coverage report.")

  # Show info where to find the report
  add_custom_command(
    TARGET ${Coverage_NAME}
    POST_BUILD
    COMMAND ;
    COMMENT "Cobertura code coverage report saved in ${Coverage_NAME}.xml.")
endfunction() # setup_target_for_coverage_gcovr_xml

# Defines a target for running and collection code coverage information
# Builds dependencies, runs the given executable and outputs reports.
# NOTE! The executable should always have a ZERO as exit code otherwise
# the coverage generation will not complete.
#
# setup_target_for_coverage_gcovr_html(
#     NAME ctest_coverage                    # New target name
#     EXECUTABLE ctest -j ${PROCESSOR_COUNT} # Executable in PROJECT_BINARY_DIR
#     DEPENDENCIES executable_target         # Dependencies to build first
#     BASE_DIRECTORY "../"                   # Base directory for report
#                                            #  (defaults to PROJECT_SOURCE_DIR)
#     EXCLUDE "src/dir1/*" "src/dir2/*"      # Patterns to exclude (can be relative
#                                            #  to BASE_DIRECTORY, with CMake 3.4+)
# )
# The user can set the variable GCOVR_ADDITIONAL_ARGS to supply additional flags to the
# GCVOR command.
function(setup_target_for_coverage_gcovr_html)

  set(options NONE)
  set(oneValueArgs BASE_DIRECTORY NAME)
  set(multiValueArgs EXCLUDE EXECUTABLE EXECUTABLE_ARGS DEPENDENCIES)
  cmake_parse_arguments(Coverage "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT GCOVR_PATH)
    message(FATAL_ERROR "gcovr not found! Aborting...")
  endif() # NOT GCOVR_PATH

  # Set base directory (as absolute path), or default to PROJECT_SOURCE_DIR
  if(DEFINED Coverage_BASE_DIRECTORY)
    get_filename_component(BASEDIR ${Coverage_BASE_DIRECTORY} ABSOLUTE)
  else()
    set(BASEDIR ${PROJECT_SOURCE_DIR})
  endif()

  # Collect excludes (CMake 3.4+: Also compute absolute paths)
  set(GCOVR_EXCLUDES "")
  foreach(EXCLUDE ${Coverage_EXCLUDE} ${COVERAGE_EXCLUDES} ${COVERAGE_GCOVR_EXCLUDES})
    if(CMAKE_VERSION VERSION_GREATER 3.4)
      get_filename_component(EXCLUDE ${EXCLUDE} ABSOLUTE BASE_DIR ${BASEDIR})
    endif()
    list(APPEND GCOVR_EXCLUDES "${EXCLUDE}")
  endforeach()
  list(REMOVE_DUPLICATES GCOVR_EXCLUDES)

  # Combine excludes to several -e arguments
  set(GCOVR_EXCLUDE_ARGS "")
  foreach(EXCLUDE ${GCOVR_EXCLUDES})
    list(APPEND GCOVR_EXCLUDE_ARGS "-e")
    list(APPEND GCOVR_EXCLUDE_ARGS "${EXCLUDE}")
  endforeach()

  # Set up commands which will be run to generate coverage data Run tests
  set(GCOVR_HTML_EXEC_TESTS_CMD ${Coverage_EXECUTABLE} ${Coverage_EXECUTABLE_ARGS})
  # Create folder
  set(GCOVR_HTML_FOLDER_CMD ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/${Coverage_NAME})
  # Running gcovr
  set(GCOVR_HTML_CMD
      ${GCOVR_PATH}
      --html
      ${Coverage_NAME}/index.html
      --html-details
      -r
      ${BASEDIR}
      ${GCOVR_ADDITIONAL_ARGS}
      ${GCOVR_EXCLUDE_ARGS}
      --object-directory=${PROJECT_BINARY_DIR})

  if(CODE_COVERAGE_VERBOSE)
    message(STATUS "Executed command report")

    message(STATUS "Command to run tests: ")
    string(REPLACE ";" " " GCOVR_HTML_EXEC_TESTS_CMD_SPACED "${GCOVR_HTML_EXEC_TESTS_CMD}")
    message(STATUS "${GCOVR_HTML_EXEC_TESTS_CMD_SPACED}")

    message(STATUS "Command to create a folder: ")
    string(REPLACE ";" " " GCOVR_HTML_FOLDER_CMD_SPACED "${GCOVR_HTML_FOLDER_CMD}")
    message(STATUS "${GCOVR_HTML_FOLDER_CMD_SPACED}")

    message(STATUS "Command to generate gcovr HTML coverage data: ")
    string(REPLACE ";" " " GCOVR_HTML_CMD_SPACED "${GCOVR_HTML_CMD}")
    message(STATUS "${GCOVR_HTML_CMD_SPACED}")
  endif()

  add_custom_target(
    ${Coverage_NAME}
    COMMAND ${GCOVR_HTML_EXEC_TESTS_CMD}
    COMMAND ${GCOVR_HTML_FOLDER_CMD}
    COMMAND ${GCOVR_HTML_CMD}
    BYPRODUCTS ${PROJECT_BINARY_DIR}/${Coverage_NAME}/index.html # report directory
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    DEPENDS ${Coverage_DEPENDENCIES}
    VERBATIM # Protect arguments to commands
    COMMENT "Running gcovr to produce HTML code coverage report.")

  # Show info where to find the report
  add_custom_command(
    TARGET ${Coverage_NAME}
    POST_BUILD
    COMMAND ;
    COMMENT "Open ./${Coverage_NAME}/index.html in your browser to view the coverage report.")

endfunction() # setup_target_for_coverage_gcovr_html

function(append_coverage_compiler_flags)
  set(CMAKE_C_FLAGS
      "${CMAKE_C_FLAGS} ${COVERAGE_COMPILER_FLAGS}"
      PARENT_SCOPE)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} ${COVERAGE_COMPILER_FLAGS}"
      PARENT_SCOPE)
  set(CMAKE_Fortran_FLAGS
      "${CMAKE_Fortran_FLAGS} ${COVERAGE_COMPILER_FLAGS}"
      PARENT_SCOPE)
  message(STATUS "Appending code coverage compiler flags: ${COVERAGE_COMPILER_FLAGS}")
endfunction() # append_coverage_compiler_flags

# Setup coverage for specific library
function(append_coverage_compiler_flags_to_target name)
  target_compile_options(${name} PRIVATE ${COVERAGE_COMPILER_FLAGS})
endfunction()
