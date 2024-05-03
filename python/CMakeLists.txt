# pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
message(STATUS "Downloading pybind11, this may take a few minutes.")
FetchContent_MakeAvailable(pybind11)

if(NOT Torch_PYTHON_BINDING)
  message(FATAL_ERROR "Could not find the libTorch Python binding")
endif()

if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
  add_custom_target(pyneml2 ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${NEML2_SOURCE_DIR}/python/neml2/__init__.py ${NEML2_BINARY_DIR}/python/neml2/__init__.py
    COMMENT "Copying __init__.py"
  )
endif()

# macro for defining a submodule with given source files
macro(add_submodule mname msrcs)
  pybind11_add_module(${mname} MODULE ${msrcs})
  set_target_properties(${mname} PROPERTIES LIBRARY_OUTPUT_DIRECTORY neml2)
  target_include_directories(${mname} PUBLIC ${NEML2_SOURCE_DIR})
  target_link_libraries(${mname} PRIVATE pybind11::headers)
  target_link_libraries(${mname} PUBLIC neml2 ${Torch_PYTHON_BINDING})
  install(TARGETS ${mname} LIBRARY DESTINATION .)
endmacro()

# macro for defining a submodule with source files automatically discovered from a given directory
macro(add_submodule_dir mname)
  file(GLOB_RECURSE msrcs neml2/${mname}/*.cxx)
  add_submodule(${mname} "${msrcs}")
endmacro()

# Actually add the submodules (using the above macros)
add_submodule_dir(base)
add_submodule_dir(tensors)
add_submodule_dir(models)
add_submodule(math neml2/misc/math.cxx)

# Install python artifacts
install(FILES neml2/__init__.py DESTINATION .)
