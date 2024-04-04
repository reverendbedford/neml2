cmake_minimum_required(VERSION 3.14)

project(NEML2 LANGUAGES CXX)

# Setup modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${NEML2_SOURCE_DIR}/cmake/Modules/")

# Ninja only: limit number of jobs in a CI build
set_property(GLOBAL PROPERTY JOB_POOLS CI=6)

# Get NEML2 dependency
include(NEML2Dependency)

# FindPython should return the first matching Python
if(POLICY CMP0094)
      cmake_policy(SET CMP0094 NEW)
endif()

# Suppress the warning related to the new policy on fetch content's timestamp
if(POLICY CMP0135)
      cmake_policy(SET CMP0135 NEW)
endif()

# Suppress the warning related to the new policy on FindPythonXXX
if(POLICY CMP0148)
      cmake_policy(SET CMP0148 NEW)
endif()

# We will rely on FetchContent to download various packages
include(FetchContent)

# Accept Release, Debug, and RelWithDebInfo, add Coverage build types
set(CMAKE_CXX_FLAGS_COVERAGE
      "-O0 -fprofile-arcs -ftest-coverage"
      CACHE STRING "Flags used by C++ compiler during coverage builds."
      FORCE)

# Set the default build type
if(EXISTS "${NEML2_SOURCE_DIR}/.git")
      set(DEFAULT_BUILD_TYPE "Debug")
else()
      set(DEFAULT_BUILD_TYPE "Release")
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
      find_package(Python COMPONENTS Interpreter Development.Module)
      find_package(pybind11 CONFIG REQUIRED HINTS ${Python_SITEARCH})

      # Figure out the torch path
      if(NOT LIBTORCH_DIR STREQUAL PYTORCH_DIR)
            message(WARNING
                  "LIBTORCH_DIR and PYTORCH_DIR are different. "
                  "This may lead to errors when importing neml2 in Python. \n"
                  "  LIBTORCH_DIR=${LIBTORCH_DIR}\n"
                  "  PYTORCH_DIR=${PYTORCH_DIR}")
      endif()

      add_subdirectory(python)
endif()
