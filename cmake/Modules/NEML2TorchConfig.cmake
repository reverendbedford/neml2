# libTorch comes with two flavors: one with cxx11 abi, one without.
# We should be consistent with whatever is detected from the libTorch.
try_compile(Torch_CXX11_ABI
  ${NEML2_SOURCE_DIR}/cmake/detect_torch_cxx11_abi/build
  ${NEML2_SOURCE_DIR}/cmake/detect_torch_cxx11_abi
  TEST
  CMAKE_FLAGS "-DLIBTORCH_DIR=${LIBTORCH_DIR}"
)

set(Torch_INCLUDE_DIRECTORIES ${LIBTORCH_DIR}/include/torch/csrc/api/include ${LIBTORCH_DIR}/include)
set(Torch_LINK_DIRECTORIES ${LIBTORCH_DIR}/lib)

file(GLOB _C10LIBS ${LIBTORCH_DIR}/lib/libc10*)
file(GLOB _TORCHLIBS ${LIBTORCH_DIR}/lib/libtorch*)
set(Torch_LIBRARIES ${_C10LIBS} ${_TORCHLIBS})
find_library(Torch_PYTHON_BINDING torch_python HINTS ${Torch_LINK_DIRECTORIES})
list(REMOVE_ITEM Torch_LIBRARIES ${Torch_PYTHON_BINDING})

# Install rpath, important for a relocatable install
if(NEML2_WHEELS)
  if(UNIX)
    if(APPLE)
      set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/lib;@loader_path/../torch/lib;@loader_path/../../torch/lib")
    else()
      set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/lib;$ORIGIN/../torch/lib;$ORIGIN/../../torch/lib")
    endif()
  endif()
else()
  if(UNIX)
    if(APPLE)
      set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/lib;${Torch_LINK_DIRECTORIES}")
    else()
      set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/lib;${Torch_LINK_DIRECTORIES}")
    endif()
  endif()
endif()
