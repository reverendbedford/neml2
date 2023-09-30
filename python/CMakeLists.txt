file(GLOB_RECURSE SRCS *.cxx)
pybind11_add_module(pyneml2 ${SRCS})
target_include_directories(pyneml2 PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Linking against libtorch_python is necessary because it defines necessary
# type casters but is not inclulded in TORCH_LIBRARIES
target_link_libraries(pyneml2 PRIVATE neml2 ${LIBTORCH_DIR}/lib/libtorch_python.so)

# I want the python module name to be neml2 instead of pyneml2...
set_target_properties(pyneml2 PROPERTIES OUTPUT_NAME neml2)
