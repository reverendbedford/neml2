find_program(SPHINX_EXECUTABLE NAMES sphinx-build
      DOC "Sphinx executable"
      )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sphinx DEFAULT_MSG SPHINX_EXECUTABLE)

mark_as_advanced(SPHINX_EXECUTABLE Sphinx_DIR)
