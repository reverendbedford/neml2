if(NOT LIBTORCH_PYTHON)
  message(FATAL_ERROR "Could not find the libTorch Python binding")
endif()

message(STATUS "Found libTorch Python binding: ${LIBTORCH_PYTHON}")

file(GLOB_RECURSE SRCS *.cxx)
pybind11_add_module(pyneml2 ${SRCS})
target_include_directories(pyneml2 PUBLIC ${NEML2_SOURCE_DIR}/python ${NEML2_BINARY_DIR}/python)
target_link_libraries(pyneml2 PUBLIC neml2 ${LIBTORCH_PYTHON})
set_target_properties(pyneml2 PROPERTIES OUTPUT_NAME neml2)

list(APPEND PYNEML2_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
list(APPEND PYNEML2_INSTALL_RPATH ${LIBTORCH_DIR}/lib)
set_target_properties(pyneml2 PROPERTIES INSTALL_RPATH "${PYNEML2_INSTALL_RPATH}")
install(TARGETS pyneml2 LIBRARY DESTINATION lib)

# Binary Python modules (such as pybind11 bindings) don't work well with
# PyLance, though other language servers may work. This is by design mainly due
# to security considerations (see e.g. https://github.com/microsoft/pylance-release/issues/70).
# To provide language servers with hints on modules and types, we use
# pybind11-stubgen to generate stub type information.
add_custom_target(pyneml2-stub
  DEPENDS pyneml2
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/python
  COMMAND PYTHONPATH=. pybind11-stubgen -o . neml2
  COMMENT "Generating .pyi files from NEML2 Python binding"
)
