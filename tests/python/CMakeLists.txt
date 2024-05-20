# ----------------------------------------------------------------------------
# Dependencies and 3rd party packages
# ----------------------------------------------------------------------------
find_package(Python COMPONENTS Interpreter)
execute_process(
  COMMAND_ERROR_IS_FATAL ANY
  COMMAND ${Python_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/check_python_dep.py ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
)

# ----------------------------------------------------------------------------
# Install test resources
# ----------------------------------------------------------------------------
if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
  file(CREATE_LINK ${NEML2_SOURCE_DIR}/tests/python ${NEML2_BINARY_DIR}/tests/python SYMBOLIC)
endif()

install(DIRECTORY .
  TYPE BIN
  FILES_MATCHING
  PATTERN "*.py"
)
