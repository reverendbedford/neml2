# libTorch comes with two flavors: one with cxx11 abi, one without.
# We should be consistent with whatever is detected from the libTorch.
try_compile(Torch_CXX11_ABI
  ${NEML2_BINARY_DIR}/cmake/detect_torch_cxx11_abi/build
  ${NEML2_SOURCE_DIR}/cmake/detect_torch_cxx11_abi
  TEST
  CMAKE_FLAGS "-DLIBTORCH_DIR=${LIBTORCH_DIR}"
)

if(Torch_CXX11_ABI)
  set(GLIBCXX_USE_CXX11_ABI 1)
else()
  set(GLIBCXX_USE_CXX11_ABI 0)
endif()

set(Torch_INCLUDE_DIRECTORIES ${LIBTORCH_DIR}/include/torch/csrc/api/include ${LIBTORCH_DIR}/include)
set(Torch_LINK_DIRECTORIES ${LIBTORCH_DIR}/lib)

file(GLOB _C10LIBS ${LIBTORCH_DIR}/lib/libc10*)
file(GLOB _TORCHLIBS ${LIBTORCH_DIR}/lib/libtorch*)
set(Torch_LIBRARIES ${_C10LIBS} ${_TORCHLIBS})
find_library(Torch_PYTHON_BINDING torch_python HINTS ${Torch_LINK_DIRECTORIES})
list(REMOVE_ITEM Torch_LIBRARIES ${Torch_PYTHON_BINDING})
