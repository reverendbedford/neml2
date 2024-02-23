find_library(TORCH_PYTHON_LIB torch_python HINTS ${LIBTORCH_DIR}/lib)

if(NOT TORCH_PYTHON_LIB)
  message(FATAL_ERROR "Could not find the libTorch Python binding")
endif()

message(STATUS "Found libTorch Python binding: ${TORCH_PYTHON_LIB}")

file(GLOB_RECURSE SRCS *.cxx)
pybind11_add_module(pyneml2 ${SRCS})
target_include_directories(pyneml2 PUBLIC ${NEML2_SOURCE_DIR}/python ${NEML2_BINARY_DIR}/python)
target_link_libraries(pyneml2 PUBLIC neml2 ${TORCH_PYTHON_LIB})
set_target_properties(pyneml2 PROPERTIES OUTPUT_NAME neml2)

# Binary Python modules (such as pybind11 bindings) don't work well with
# PyLance, though other language servers may work. This is by design mainly due
# to security considerations (see e.g. https://github.com/microsoft/pylance-release/issues/70).
# To provide language servers with hints on modules and types, we use
# pybind11-stubgen to generate stub type information.
add_custom_target(pyneml2-stub
  DEPENDS pyneml2
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/python
  COMMAND PYTHONPATH=. pybind11-stubgen --ignore-all-errors -o . neml2
  COMMENT "Generating .pyi files from NEML2 Python binding"
)
