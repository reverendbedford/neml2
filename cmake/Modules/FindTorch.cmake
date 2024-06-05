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
  message(STATUS "Configuring Torch")
  FetchContent_MakeAvailable(torch)
  set(LIBTORCH_DIR ${torch_SOURCE_DIR})
endif()

# At this point, if LIBTORCH_DIR is still not set, then both plan A and plan B have failed :(
if(NOT DEFINED LIBTORCH_DIR)
  message(FATAL_ERROR
    "LIBTORCH_DIR is not set. Please refer to https://reverendbedford.github.io/neml2/install.html for more information.")
else()
  message(STATUS "Using libTorch at: ${LIBTORCH_DIR}")
  include(NEML2TorchConfig)
endif()
