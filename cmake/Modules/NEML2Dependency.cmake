# ###################################################
# Find and set NEML2 metadata
#
# Variables:
# NEML2_PYTORCH_VERSION_MAJOR: The major version number of the compatible pytorch
# NEML2_PYTORCH_VERSION_MINOR: The major version number of the compatible pytorch
# NEML2_PYTORCH_VERSION_PATCH: The major version number of the compatible pytorch
# ###################################################

file(READ ${NEML2_SOURCE_DIR}/DEPENDENCY.json NEML2_DEPENDENCY)

string(JSON NEML2_PYTORCH_VERSION_MAJOR GET ${NEML2_DEPENDENCY} pytorch major)
string(JSON NEML2_PYTORCH_VERSION_MINOR GET ${NEML2_DEPENDENCY} pytorch minor)
string(JSON NEML2_PYTORCH_VERSION_PATCH GET ${NEML2_DEPENDENCY} pytorch patch)

if(${NEML2_PYTORCH_VERSION_MAJOR} AND ${NEML2_PYTORCH_VERSION_MINOR} AND ${NEML2_PYTORCH_VERSION_PATCH})
  list(APPEND NEML2_PYTORCH_VERSION_LIST ${NEML2_PYTORCH_VERSION_MAJOR} ${NEML2_PYTORCH_VERSION_MINOR} ${NEML2_PYTORCH_VERSION_PATCH})
  list(JOIN NEML2_PYTORCH_VERSION_LIST "." NEML2_PYTORCH_VERSION)
  message(STATUS "Current compatible PyTorch version: ${NEML2_PYTORCH_VERSION}")
else()
  message(FATAL_ERROR "Failed to detect current compatible PyTorch version")
endif()
