if(NOT DEFINED DOXYGEN_EXECUTABLE)
  if(UNIX)
    set(FETCHCONTENT_QUIET FALSE)
    FetchContent_Declare(
      doxygen
      URL https://github.com/doxygen/doxygen/releases/download/Release_1_9_8/doxygen-1.9.8.linux.bin.tar.gz
    )
    FetchContent_MakeAvailable(doxygen)
    set(DOXYGEN_EXECUTABLE ${doxygen_SOURCE_DIR}/bin/doxygen)
  else()
    message(FATAL_ERROR "We only download a default doxygen on linux for CI purposes. For other operating systems, please specify DOXYGEN_EXECUTABLE.")
  endif()
endif()

find_package(Python COMPONENTS Interpreter)

# Generate Doxyfile for HTML
message(STATUS "Generating Doxyfile for HTML")
file(WRITE ${CMAKE_BINARY_DIR}/doc/DoxyfileHTML.in "")
file(READ ${PROJECT_SOURCE_DIR}/doc/config/Doxyfile.in CONTENTS)
file(APPEND ${CMAKE_BINARY_DIR}/doc/DoxyfileHTML.in ${CONTENTS})
file(READ ${PROJECT_SOURCE_DIR}/doc/config/HTML.in CONTENTS)
file(APPEND ${CMAKE_BINARY_DIR}/doc/DoxyfileHTML.in ${CONTENTS})
configure_file(
  ${CMAKE_BINARY_DIR}/doc/DoxyfileHTML.in
  ${CMAKE_BINARY_DIR}/doc/DoxyfileHTML.sh
)

# Generate Doxyfile for LATEX
message(STATUS "Generating Doxyfile for LATEX")
file(WRITE ${CMAKE_BINARY_DIR}/doc/DoxyfileLATEX.in "")
file(READ ${PROJECT_SOURCE_DIR}/doc/config/Doxyfile.in CONTENTS)
file(APPEND ${CMAKE_BINARY_DIR}/doc/DoxyfileLATEX.in ${CONTENTS})
file(READ ${PROJECT_SOURCE_DIR}/doc/config/LATEX.in CONTENTS)
file(APPEND ${CMAKE_BINARY_DIR}/doc/DoxyfileLATEX.in ${CONTENTS})
configure_file(
  ${CMAKE_BINARY_DIR}/doc/DoxyfileLATEX.in
  ${CMAKE_BINARY_DIR}/doc/DoxyfileLATEX.sh
)

add_custom_target(doc-syntax
  DEPENDS unit_tests
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc
  COMMAND ${CMAKE_BINARY_DIR}/tests/unit_tests Registry -c syntax -r compact
  COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/syntax_to_md.py syntax.yml content/_99_syntax.md
  COMMENT "Generate NEML2 syntax and convert to markdown"
  VERBATIM
)

add_custom_target(doc-html
  DEPENDS doc-syntax
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc
  COMMAND ${DOXYGEN_EXECUTABLE} -q DoxyfileHTML.sh
  COMMENT "Generate Doxygen HTML"
  VERBATIM
)

add_custom_target(doc-latex
  DEPENDS doc-syntax
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc
  COMMAND ${DOXYGEN_EXECUTABLE} -q DoxyfileLATEX.sh
  COMMENT "Generate Doxygen LATEX"
  VERBATIM
)

add_custom_target(doc-pdf
  DEPENDS doc-latex
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc/build/latex
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/doc/content/asset ${CMAKE_BINARY_DIR}/doc/build/latex
  COMMAND latexmk -pdf refman.tex -outdir=${CMAKE_BINARY_DIR}/doc/build/pdf -silent
  COMMENT "Generate Doxygen PDF"
  VERBATIM
)
