option(DOXYGEN "Build doxygen documentation" ON)

if(DOXYGEN)
  find_package(Doxygen REQUIRED)

  add_custom_target(doc-syntax
    DEPENDS unit_tests
    COMMAND ${CMAKE_BINARY_DIR}/tests/unit_tests Registry -c syntax
    COMMAND python TexSyntax.py doc/_05_syntax.md
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc/doxygen
  )

  add_custom_target(doc-html
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/doc/doxygen/Doxyfile_html
    COMMAND cp -r doc/asset ${CMAKE_BINARY_DIR}/doc/class-doc/html
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc/doxygen
    COMMENT "Generate Doxygen HTML"
    VERBATIM)

  add_custom_target(doc-latex
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/doc/doxygen/Doxyfile_latex
    COMMAND cp -r doc/asset ${CMAKE_BINARY_DIR}/doc/class-doc/latex
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc/doxygen
    COMMENT "Generate Doxygen LATEX"
    VERBATIM)

  add_custom_target(doc-pdf
    DEPENDS doc-latex
    COMMAND latexmk -pdf refman.tex
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc/class-doc/latex
    COMMENT "Generate Doxygen PDF"
    VERBATIM)
endif()
