cmake_minimum_required(VERSION 3.14)

project(NEML2 LANGUAGES CXX)

# ## Setup modules ###
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${NEML2_SOURCE_DIR}/cmake/Modules/")

# Accept Release, Debug, and RelWithDebInfo, add Coverage build types
set(CMAKE_CXX_FLAGS_COVERAGE
      "-O0 -fprofile-arcs -ftest-coverage"
      CACHE STRING "Flags used by C++ compiler during coverage builds."
      FORCE)

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(EXISTS "${NEML2_SOURCE_DIR}/.git")
      set(DEFAULT_BUILD_TYPE "Debug")
endif()

if(NOT CMAKE_BUILD_TYPE)
      message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
      set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
            STRING "Choose the type of build." FORCE)

      # Set the possible values of build type for cmake-gui
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "Coverage")
endif()

# Add the unity option to the cache
option(CMAKE_UNITY_BUILD "Use a unity build" OFF)

# Enable testing
enable_testing()

# c++ 17 support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Suppress the warning related to the new policy on fetch content's timestamp
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
      cmake_policy(SET CMP0135 NEW)
endif()

# Suppress the warning related to the new policy on FindPythonXXX
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27.0")
      cmake_policy(SET CMP0148 NEW)
endif()

# If LIBTORCH_DIR is not defined, we should spend some effort to provide a sensible default.
# Plan A: If we can find Python, and PyTorch is available as a python package,
# we can just use the libTorch shipped together with the PyTorch package.
# This is actually preferred when we build Python bindings for NEML2.
if(NOT DEFINED LIBTORCH_DIR)
      message(STATUS "LIBTORCH_DIR not set, try to find a PyTorch installation.")

      find_package(Python COMPONENTS Interpreter)

      if(Python_Interpreter_FOUND)
            # Figure out the torch path
            execute_process(
                  COMMAND "${Python_EXECUTABLE}" "scripts/find_torch.py"
                  OUTPUT_VARIABLE PYTORCH_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE
            )

            if(NOT ${PYTORCH_DIR} STREQUAL "")
                  set(LIBTORCH_DIR ${PYTORCH_DIR})
                  message(STATUS "Using libTorch inside the autodected PyTorch direcotry: ${LIBTORCH_DIR}")
            endif()
      endif()
endif()

# Plan B: If we are on Unix systems, we could default to downloading a CPU-only
# libTorch.
if(NOT DEFINED LIBTORCH_DIR)
      message(STATUS "LIBTORCH_DIR not set, try to download a libTorch (CPU) from the official website.")

      if(UNIX)
            if(NOT APPLE)
                  FetchContent_Declare(torch URL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip)
            else()
                  if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
                        set(APPLE_SILICON ON)

                        # This URL will need to be maintained
                        FetchContent_Declare(torch URL https://files.pythonhosted.org/packages/7b/7c/4d8728e6f8dbe2b8af054bd92c290d94c633270443514e3ee4b768125cf9/torch-2.1.0-cp311-none-macosx_11_0_arm64.whl)
                  else()
                        set(APPLE_SILICON OFF)
                        FetchContent_Declare(torch URL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip)
                  endif()

                  message(WARNING ${CMAKE_SYSTEM_PROCESSOR})
            endif()

            message(STATUS "Downloading libTorch, this may take a few minutes.")
            FetchContent_MakeAvailable(torch)

            if(APPLE AND APPLE_SILICON)
                  set(LIBTORCH_DIR ${torch_SOURCE_DIR}/torch)
            else()
                  set(LIBTORCH_DIR ${torch_SOURCE_DIR})
            endif()

            message(STATUS "Using downloaded libTorch: ${LIBTORCH_DIR}")
      else()
            message(STATUS "We only download a default libTorch (CPU) on Linux and macOS.")
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
endif()

find_package(Torch REQUIRED HINTS "${LIBTORCH_DIR}/share/cmake")

# libTorch comes with two flavors: one with cxx11 abi, one without.
# We should be consistent with whatever is detected from the libTorch.
if(TORCH_CXX_FLAGS MATCHES "-D_GLIBCXX_USE_CXX11_ABI=1")
      add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
      message(STATUS "NEML2 is using CXX11 ABI to be consistent with libTorch")
else()
      add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
      message(STATUS "NEML2 is using pre-CXX11 ABI to be consistent with libTorch")
endif()

# Catch2, for testing
add_subdirectory("${NEML2_SOURCE_DIR}/extern/Catch2" "${NEML2_BINARY_DIR}/extern/Catch2")
list(APPEND CMAKE_MODULE_PATH "${NEML2_SOURCE_DIR}/extern/Catch2/contrib")
include(CTest)
include(Catch)

if(BUILD_TESTING)
      add_subdirectory(tests)
endif()

# base library
add_subdirectory(src/neml2)

# hit for parsing
add_subdirectory("${NEML2_SOURCE_DIR}/extern/hit" "${NEML2_BINARY_DIR}/extern/hit")

# Doxygen
option(NEML2_DOC "Build NEML2 documentation: doxygen" OFF)

if(NEML2_DOC)
      add_subdirectory(doc)
endif()

# Python bindings
option(NEML2_PYBIND "Build python bindings" OFF)

if(NEML2_PYBIND)
      find_package(Python COMPONENTS Interpreter Development)

      # Figure out the torch path
      execute_process(
            COMMAND "${Python_EXECUTABLE}" "scripts/find_torch.py"
            OUTPUT_VARIABLE PYTORCH_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
      )

      if(NOT ${PYTORCH_DIR} STREQUAL "")
            if(NOT LIBTORCH_DIR STREQUAL PYTORCH_DIR)
                  message(WARNING
                        "LIBTORCH_DIR and PYTORCH_DIR are different. "
                        "This may lead to errors when importing neml2 in Python. \n"
                        "  LIBTORCH_DIR=${LIBTORCH_DIR}\n"
                        "  PYTORCH_DIR=${PYTORCH_DIR}")
            endif()
      endif()

      add_subdirectory(${NEML2_SOURCE_DIR}/extern/pybind11)

      add_subdirectory(python)
endif()
