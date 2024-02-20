include(NEML2Artifacts)

if(NOT LIBTORCH_PYTHON)
  message(FATAL_ERROR "Could not find the libTorch Python binding")
endif()

# macro for defining a submodule
macro(add_submodule mname)
  file(GLOB_RECURSE msrcs neml2/${mname}/*.cxx)
  python_add_library(${mname} MODULE ${msrcs} WITH_SOABI)
  target_include_directories(${mname} PUBLIC ${NEML2_SOURCE_DIR})
  target_link_libraries(${mname} PRIVATE pybind11::headers)
  target_link_libraries(${mname} PUBLIC neml2 ${LIBTORCH_PYTHON})
  set_target_properties(${mname} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NEML2_BINARY_DIR}/python/neml2)

  # Binary Python modules (such as pybind11 bindings) don't work well with
  # PyLance, though other language servers may work. This is by design mainly due
  # to security considerations (see e.g. https://github.com/microsoft/pylance-release/issues/70).
  # To provide language servers with hints on modules and types, we use
  # pybind11-stubgen to generate stub type information.
  # add_custom_command(TARGET ${mname}
  # POST_BUILD
  # WORKING_DIRECTORY ${NEML2_BINARY_DIR}/python
  # COMMAND PYTHONPATH=$ENV{PYTHONPATH}:. pybind11-stubgen -o . neml2.${mname}
  # COMMENT "Generating typing information for python submodule neml2.${mname}"
  # )

  # Also install the .pyi stubs
  # install(FILES ${NEML2_BINARY_DIR}/python/neml2/${mname}.pyi DESTINATION ${CMAKE_INSTALL_PREFIX}/neml2)
endmacro()

# Actually add the submodules (using the above macro)
add_submodule(tensors)
add_submodule(math)
