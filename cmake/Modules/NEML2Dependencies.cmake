include(FetchContent) # For downloading dependencies
include(ExternalProject) # TriBITs really really is only designed for standalone build

# PyTorch
if(UNIX)
  if(NOT APPLE)
    FetchContent_Declare(torch URL https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip)
  else()
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
      FetchContent_Declare(torch URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${PYTORCH_VERSION}.zip)
    else()
      FetchContent_Declare(torch URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-${PYTORCH_VERSION}.zip)
    endif()
  endif()
endif()

find_package(Torch) # This gets redirected to our FindTorch.cmake

# Doxygen for documentation
string(REPLACE "." "_" DOXYGEN_RELEASE ${DOXYGEN_VERSION})
FetchContent_Declare(
  doxygen
  URL https://github.com/doxygen/doxygen/releases/download/Release_${DOXYGEN_RELEASE}/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz
)

# Doxygen html stylesheet
FetchContent_Declare(
  doxygen-awesome-css
  GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
  GIT_TAG v${DOXYGEN_AWESOME_VERSION}
)

# Pybind11 for Python bindings
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v${PYBIND11_VERSION}
)

# WASP and HIT for parsing input files
ExternalProject_Add(
  wasp
  GIT_REPOSITORY https://code.ornl.gov/neams-workbench/wasp
  GIT_TAG ${WASP_VERSION}
  PREFIX wasp
  CMAKE_ARGS
  -DCMAKE_CXX_FLAGS:STRING=-D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}
  -DCMAKE_BUILD_TYPE:STRING=RELEASE
  -DCMAKE_INSTALL_PREFIX:STRING=${NEML2_BINARY_DIR}/wasp/install
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -Dwasp_ENABLE_ALL_PACKAGES:BOOL=OFF
  -Dwasp_ENABLE_wasphit:BOOL=ON
  -Dwasp_ENABLE_testframework:BOOL=OFF
  -Dwasp_ENABLE_TESTS:BOOL=OFF
  -DBUILD_SHARED_LIBS:BOOL=OFF
  -DDISABLE_HIT_TYPE_PROMOTION:BOOL=ON
  TEST_EXCLUDE_FROM_MAIN ON
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
)
FetchContent_Declare(
  hit
  GIT_REPOSITORY https://github.com/hugary1995/hit
  GIT_TAG ${HIT_VERSION}
)

# Catch2 for testing
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v${CATCH2_VERSION}
)

# gperftools for profiling
FetchContent_Declare(
  gperftools
  EXCLUDE_FROM_ALL
  GIT_REPOSITORY https://github.com/gperftools/gperftools.git
  GIT_TAG gperftools-${GPERFTOOLS_VERSION}
)

# C++ implementation of argparse
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v${ARGPARSE_VERSION}
)
