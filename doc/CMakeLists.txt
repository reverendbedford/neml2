option(SPHINX_HTML "Build html help with Sphinx" ON)
option(SPHINX_PDF "Build PDF manual with Sphinx" ON)
option(DOXYGEN "Build doxygen documentation" ON)

if(SPHINX_HTML OR SPHINX_PDF)
  find_package(Sphinx REQUIRED)
  set(USE_SPHINX ON)
endif()

if(USE_SPHINX OR DOXYGEN)
  find_package(Doxygen REQUIRED)

  add_custom_target(doc-doxygen
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/doc/doxygen/Doxyfile
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc/doxygen
    COMMENT "Generate Doxygen XML"
    VERBATIM)
endif()

if(SPHINX_HTML)
  set(SPHINX_HTML_DIR ${PROJECT_BINARY_DIR}/doc/html)
  set(SPHINX_SRC_DIR ${PROJECT_BINARY_DIR}/doc/sphinx)
  add_custom_target(doc-html ${SPHINX_EXECUTABLE}
    -b html
    ${SPHINX_SRC_DIR}
    ${SPHINX_HTML_DIR}
    DEPENDS doc-doxygen
  )
endif()

if(SPHINX_PDF)
  set(SPHINX_PDF_DIR ${PROJECT_BINARY_DIR}/doc/pdf)
  set(SPHINX_SRC_DIR ${PROJECT_BINARY_DIR}/doc/sphinx)
  add_custom_target(doc-pdf ${SPHINX_EXECUTABLE}
    -b latex
    ${SPHINX_SRC_DIR}
    ${SPHINX_PDF_DIR}
    DEPENDS doc-doxygen
  )
endif()
