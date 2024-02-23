# ###################################################
# Find and set NEML2 metadata
#
# Variables:
# NEML2_PYTORCH_VERSION_MAJOR: The major version number of the compatible pytorch
# NEML2_PYTORCH_VERSION_MINOR: The major version number of the compatible pytorch
# NEML2_PYTORCH_VERSION_PATCH: The major version number of the compatible pytorch
# ###################################################

execute_process(
  COMMAND ${NEML2_SOURCE_DIR}/scripts/get_release_info.py compatibility pytorch major
  OUTPUT_VARIABLE NEML2_PYTORCH_VERSION_MAJOR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND ${NEML2_SOURCE_DIR}/scripts/get_release_info.py compatibility pytorch minor
  OUTPUT_VARIABLE NEML2_PYTORCH_VERSION_MINOR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND ${NEML2_SOURCE_DIR}/scripts/get_release_info.py compatibility pytorch patch
  OUTPUT_VARIABLE NEML2_PYTORCH_VERSION_PATCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(${NEML2_PYTORCH_VERSION_MAJOR} AND ${NEML2_PYTORCH_VERSION_MINOR} AND ${NEML2_PYTORCH_VERSION_PATCH})
  list(APPEND NEML2_PYTORCH_VERSION_LIST ${NEML2_PYTORCH_VERSION_MAJOR} ${NEML2_PYTORCH_VERSION_MINOR} ${NEML2_PYTORCH_VERSION_PATCH})
  list(JOIN NEML2_PYTORCH_VERSION_LIST "." NEML2_PYTORCH_VERSION)
  message(STATUS "Current compatible PyTorch version: ${NEML2_PYTORCH_VERSION}")
else()
  message(FATAL_ERROR "Failed to detect current compatible PyTorch version")
endif()
