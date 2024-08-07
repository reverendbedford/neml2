# ----------------------------------------------------------------------------
# Dependencies and 3rd party packages
# ----------------------------------------------------------------------------
message(STATUS "Configuring Doxygen")
FetchContent_MakeAvailable(doxygen)
set(DOXYGEN_EXECUTABLE ${doxygen_SOURCE_DIR}/bin/doxygen)
FetchContent_MakeAvailable(doxygen-awesome-css)

find_package(Python COMPONENTS Interpreter)
execute_process(
  COMMAND_ERROR_IS_FATAL ANY
  COMMAND ${Python_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/check_python_dep.py ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
)

# ----------------------------------------------------------------------------
# Macro for generating and configuring Doxyfile
# ----------------------------------------------------------------------------
macro(generate_doxyfile output inputs)
  file(WRITE ${output}.in "")

  foreach(input ${inputs})
    file(READ ${input} _content)
    file(APPEND ${output}.in ${_content})
  endforeach()

  configure_file(${output}.in ${output}.sh)

  file(REMOVE ${output}.in)
endmacro()

# ----------------------------------------------------------------------------
# Extract all input file syntax
# ----------------------------------------------------------------------------
add_executable(syntax-cpp-exe syntax.cxx)
target_link_libraries(syntax-cpp-exe PRIVATE neml2)
add_custom_target(syntax-cpp
  DEPENDS syntax-cpp-exe
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/doc
  COMMAND ${NEML2_BINARY_DIR}/doc/syntax-cpp-exe
  COMMAND ${Python_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/syntax_to_md.py syntax.yml content/syntax syntax_error.log
  VERBATIM
)

# ----------------------------------------------------------------------------
# Extract all Python API
# ----------------------------------------------------------------------------
add_custom_target(syntax-python
  DEPENDS base math tensors
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/python
  COMMAND ${CMAKE_COMMAND} -E make_directory ${NEML2_BINARY_DIR}/doc/content/python
  COMMAND PYTHONPATH=. pybind11-stubgen -o ${NEML2_BINARY_DIR}/doc/content/python neml2
  COMMAND ${Python_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/fixup_pystub.py ${NEML2_BINARY_DIR}/doc/content/python/neml2
  VERBATIM
)

# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------
generate_doxyfile(${NEML2_BINARY_DIR}/doc/DoxyfileHTML "config/Doxyfile.in;config/HTML.in")
generate_doxyfile(${NEML2_BINARY_DIR}/doc/DoxyfilePython "config/Doxyfile.in;config/HTML.in;config/Python.in")
add_custom_target(html ALL
  DEPENDS syntax-cpp syntax-python
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/doc
  COMMAND ${DOXYGEN_EXECUTABLE} -q DoxyfileHTML.sh
  COMMAND ${DOXYGEN_EXECUTABLE} -q DoxyfilePython.sh
  VERBATIM
)

# ----------------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------------
generate_doxyfile(${NEML2_BINARY_DIR}/doc/DoxyfileLaTeX "config/Doxyfile.in;config/LaTeX.in")
add_custom_target(latex
  DEPENDS syntax-cpp
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/doc
  COMMAND ${DOXYGEN_EXECUTABLE} -q DoxyfileLaTeX.sh
  VERBATIM
)

# ----------------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------------
add_custom_target(pdf
  DEPENDS latex
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/doc/build/latex
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${NEML2_SOURCE_DIR}/doc/content/asset ${NEML2_BINARY_DIR}/doc/build/latex
  COMMAND latexmk -pdf refman.tex -outdir=${NEML2_BINARY_DIR}/doc/build/pdf -silent
  VERBATIM
)
