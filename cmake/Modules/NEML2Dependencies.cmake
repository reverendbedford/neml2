include(FetchContent) # For downloading dependencies

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
FetchContent_Declare(
  wasp
  EXCLUDE_FROM_ALL
  GIT_REPOSITORY https://code.ornl.gov/neams-workbench/wasp
  GIT_TAG ${WASP_VERSION}
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
  GIT_REPOSITORY https://github.com/gperftools/gperftools.git
  GIT_TAG gperftools-${GPERFTOOLS_VERSION}
)

# C++ implementation of argparse
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v${ARGPARSE_VERSION}
)
