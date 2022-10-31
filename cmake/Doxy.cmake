# From https://stackoverflow.com/questions/60955881/cmake-for-doxygen

find_package(Doxygen)
if(DOXYGEN_FOUND)
  # set input and output files
  set(DOXYGEN_IN ${PROJECT_SOURCE_DIR}/doc/Doxyfile.in)
  set(DOXYGEN_OUT ${PROJECT_SOURCE_DIR}/Doxyfile)

  # request to configure the file
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

  # note the option ALL which allows to build the docs together with the application
  add_custom_target(
    docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)

  add_custom_command(
    TARGET docs
    POST_BUILD
    COMMAND mkdir
    -p
    ${PROJECT_SOURCE_DIR}/doc/html/doc
    COMMAND cp
    -r
    ${PROJECT_SOURCE_DIR}/doc/images
    ${PROJECT_SOURCE_DIR}/doc/html/doc
    COMMAND sed
    -i
    "'s|&lt;picture&gt; &lt;source media=\"(prefers-color-scheme: dark)\" srcset=\"doc/images/performance-white.png\"&gt;||g'"
    ${PROJECT_SOURCE_DIR}/doc/html/index.html
    COMMAND sed
    -i
    "\"s|&lt;/picture&gt;||g\""
    ${PROJECT_SOURCE_DIR}/doc/html/index.html
  )
else()
  message("${BoldYellow}Doxygen need to be installed to generate the doxygen documentation!${ColourReset}")
endif()
