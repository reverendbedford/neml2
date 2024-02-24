if(NOT LIBTORCH_PYTHON)
  message(FATAL_ERROR "Could not find the libTorch Python binding")
endif()

message(STATUS "Found libTorch Python binding: ${LIBTORCH_PYTHON}")

list(APPEND PYNEML2_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/neml2/lib)
list(APPEND PYNEML2_INSTALL_RPATH ${LIBTORCH_DIR}/lib)

# Define submodule tensors
file(GLOB_RECURSE tensors_srcs neml2_wrap/tensors/*.cxx)
pybind11_add_module(tensors ${tensors_srcs})
target_include_directories(tensors PUBLIC ${NEML2_SOURCE_DIR}/python ${NEML2_BINARY_DIR}/python)
target_link_libraries(tensors PUBLIC neml2 ${LIBTORCH_PYTHON})
set_target_properties(tensors PROPERTIES INSTALL_RPATH "${PYNEML2_INSTALL_RPATH}")
install(TARGETS tensors LIBRARY DESTINATION neml2)

# Define submodule math
pybind11_add_module(math neml2_wrap/misc/math.cxx)
target_include_directories(math PUBLIC ${NEML2_SOURCE_DIR}/python ${NEML2_BINARY_DIR}/python)
target_link_libraries(math PUBLIC neml2 ${LIBTORCH_PYTHON})
set_target_properties(math PROPERTIES INSTALL_RPATH "${PYNEML2_INSTALL_RPATH}")
install(TARGETS math LIBRARY DESTINATION neml2)

# Install python files
file(GLOB_RECURSE python_srcs neml2_wrap/*.py)

foreach(src ${python_srcs})
  file(RELATIVE_PATH ipath ${NEML2_SOURCE_DIR}/python/neml2_wrap ${src})
  cmake_path(GET ipath PARENT_PATH idir)
  install(FILES ${src} DESTINATION ${CMAKE_INSTALL_PREFIX}/neml2/${idir})
endforeach()

# Binary Python modules (such as pybind11 bindings) don't work well with
# PyLance, though other language servers may work. This is by design mainly due
# to security considerations (see e.g. https://github.com/microsoft/pylance-release/issues/70).
# To provide language servers with hints on modules and types, we use
# pybind11-stubgen to generate stub type information.
# add_custom_target(pyneml2-stub
# DEPENDS pyneml2
# WORKING_DIRECTORY ${NEML2_BINARY_DIR}/python
# COMMAND PYTHONPATH=. pybind11-stubgen -o . neml2
# COMMENT "Generating .pyi files from NEML2 Python binding"
# )
