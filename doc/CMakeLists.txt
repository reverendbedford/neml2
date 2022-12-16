option(DOXYGEN "Build doxygen documentation" ON)

if(DOXYGEN)
  find_package(Doxygen REQUIRED)

  add_custom_target(doc-doxygen
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/doc/doxygen/Doxyfile
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc/doxygen
    COMMENT "Generate Doxygen XML"
    VERBATIM)
endif()
