find_package(Python COMPONENTS Interpreter Development.Module)

# find_package(pybind11 CONFIG REQUIRED HINTS ${Python_SITEARCH})
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

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
  nanobind_add_module(${mname} ${msrcs})
  target_include_directories(${mname} PUBLIC ${NEML2_SOURCE_DIR})
  target_link_libraries(${mname} PUBLIC neml2 ${Torch_PYTHON_BINDING})
  install(TARGETS ${mname} LIBRARY DESTINATION .)
endmacro()

# Actually add the submodules (using the above macro)
add_submodule(tensors)
add_submodule(math)
install(DIRECTORY neml2/
  DESTINATION .
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.pyi"
)
