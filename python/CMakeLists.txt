find_package(Python COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED HINTS ${Python_SITEARCH})

if(NOT Torch_PYTHON_BINDING)
  message(FATAL_ERROR "Could not find the libTorch Python binding")
endif()

add_custom_target(pyneml2 ALL
  COMMAND ${CMAKE_COMMAND} -E copy ${NEML2_SOURCE_DIR}/python/neml2/__init__.py ${NEML2_BINARY_DIR}/python/neml2/__init__.py
  COMMENT "Copying __init__.py"
)

# macro for defining a submodule
macro(add_submodule mname)
  file(GLOB_RECURSE msrcs neml2/${mname}/*.cxx)
  python_add_library(${mname} MODULE ${msrcs} WITH_SOABI)
  target_include_directories(${mname} PUBLIC ${NEML2_SOURCE_DIR})
  target_link_libraries(${mname} PRIVATE pybind11::headers)
  target_link_libraries(${mname} PUBLIC neml2 ${Torch_PYTHON_BINDING})
  set_target_properties(${mname}
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${NEML2_BINARY_DIR}/python/neml2
    INSTALL_RPATH "$ORIGIN;$ORIGIN/lib;${Torch_LINK_DIRECTORIES}"
  )
  install(TARGETS ${mname} LIBRARY DESTINATION .)

  # Binary Python modules (such as pybind11 bindings) don't work well with
  # PyLance, though other language servers may work. This is by design mainly due
  # to security considerations (see e.g. https://github.com/microsoft/pylance-release/issues/70).
  # To provide language servers with hints on modules and types, we use
  # pybind11-stubgen to generate stub type information.
  add_custom_command(TARGET ${mname}
    POST_BUILD
    WORKING_DIRECTORY ${NEML2_BINARY_DIR}/python
    COMMAND PYTHONPATH=$ENV{PYTHONPATH}:. pybind11-stubgen -o . neml2.${mname}
    COMMENT "Generating typing information for python submodule neml2.${mname}"
  )
  install(FILES ${NEML2_BINARY_DIR}/python/neml2/${mname}.pyi DESTINATION .)
endmacro()

# Actually add the submodules (using the above macro)
# add_submodule(tensors)
add_submodule(math)
install(DIRECTORY neml2/
  DESTINATION .
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.pyi"
)
