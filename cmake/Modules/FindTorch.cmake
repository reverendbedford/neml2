# We will rely on FetchContent to download libTorch
include(FetchContent)

# If LIBTORCH_DIR is not defined, we should make some effort to provide a sensible default.
# I have 2 plans below...

# Plan A: If we can find Python, and PyTorch is available as a python package,
# we can just use the libTorch shipped together with the PyTorch package.
# This is preferred when we build Python bindings for NEML2, and so I will make
# this plan a higher priority.
if(NOT DEFINED LIBTORCH_DIR)
  find_package(Python)

  if(Python_Interpreter_FOUND)
    set(PYTORCH_DIR ${Python_SITEARCH}/torch)

    if(EXISTS ${PYTORCH_DIR})
      set(LIBTORCH_DIR ${Python_SITEARCH}/torch)
    endif()
  endif()
endif()

# Plan B: If we are on Unix systems, we could default to downloading a CPU-only
# libTorch.
if(NOT DEFINED LIBTORCH_DIR)
  if(UNIX)
    if(NOT APPLE)
      FetchContent_Declare(torch URL https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${NEML2_PYTORCH_VERSION}%2Bcpu.zip)
    else()
      if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
        set(APPLE_SILICON ON)
        FetchContent_Declare(torch URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${NEML2_PYTORCH_VERSION}.zip)
      else()
        set(APPLE_SILICON OFF)
        FetchContent_Declare(torch URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-${NEML2_PYTORCH_VERSION}.zip)
      endif()
    endif()

    message(STATUS "Downloading libTorch, this may take a few minutes.")
    FetchContent_MakeAvailable(torch)

    if(NOT torch_SOURCE_DIR)
      message(FATAL_ERROR "Failed to donwload libTorch")
    else()
      set(LIBTORCH_DIR ${torch_SOURCE_DIR})
    endif()

  else()
    message(STATUS "We only download a default libTorch (CPU) on Unix systems. This is not a Unix system.")
  endif()
endif()

# At this point, if LIBTORCH_DIR is still not set, then both plan A and plan B have failed :(
if(NOT DEFINED LIBTORCH_DIR)
  message(FATAL_ERROR
    "LIBTORCH_DIR is not set, and we could not find/download a compatible libTorch. "
    "There are two ways to fix this error:\n"
    "  1. Manually download libTorch and set LIBTORCH_DIR while running cmake.\n"
    "  2. Install Python and the PyTorch package.\n"
    "If you are on a Unix-based system and ran into this error, please submit a bug report.")
else()
  message(STATUS "Using libTorch at: ${LIBTORCH_DIR}")
  include(NEML2TorchConfig)
endif()
