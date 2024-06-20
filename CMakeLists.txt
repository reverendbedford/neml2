# ----------------------------------------------------------------------------
# Project metadata
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.23)
project(NEML2 VERSION 1.4.0 LANGUAGES CXX)

# ----------------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Project-level settings, options, and flags
# ----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${NEML2_SOURCE_DIR}/cmake/Modules) # CMake modules and macros
set(CMAKE_CXX_FLAGS_COVERAGE "-O0 -fprofile-arcs -ftest-coverage" CACHE STRING "Flags used by C++ compiler during coverage builds." FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(NEML2_TESTS "Build NEML2 tests" ON)
option(NEML2_RUNNER "Build a simple runner for benchmarking, profiling, debugging, etc." OFF)
option(NEML2_PYBIND "Build NEML2 Python bindings" OFF)
option(NEML2_DOC "Build NEML2 documentation (html)" OFF)

# ----------------------------------------------------------------------------
# Dependencies and 3rd party packages
# ----------------------------------------------------------------------------
set(PYTORCH_VERSION "2.2.2")
set(DOXYGEN_VERSION "1.10.0")
set(DOXYGEN_AWESOME_VERSION "2.3.2")
set(PYBIND11_VERSION "2.12.0")
set(WASP_VERSION "c8c9ce4259115973f147e345608450d87dc390c4")
set(HIT_VERSION "6ad08fe9c6d58db94543fe70b79cea11609aec22")
set(CATCH2_VERSION "3.5.4")
set(GPERFTOOLS_VERSION "2.15")
set(ARGPARSE_VERSION "3.0")
include(NEML2Dependencies)
find_package(Torch) # This gets redirected to our FindTorch.cmake

# ----------------------------------------------------------------------------
# PyTorch ships libraries with or without CXX11 ABI
# ----------------------------------------------------------------------------
if(Torch_CXX11_ABI)
      add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
else()
      add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

# ----------------------------------------------------------------------------
# For relocatable install
# ----------------------------------------------------------------------------
if(UNIX AND APPLE)
      set(EXEC_DIR "@loader_path")
elseif(UNIX AND NOT APPLE)
      set(EXEC_DIR "$ORIGIN")
endif()

# ----------------------------------------------------------------------------
# Build types
# ----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "Coverage")
endif()

# ----------------------------------------------------------------------------
# Subdirectories
# ----------------------------------------------------------------------------
# base neml2 library
add_subdirectory(src/neml2)

# tests
if(NEML2_TESTS)
      add_subdirectory(tests)
endif()

# runner
if(NEML2_RUNNER)
      add_subdirectory(runner)
endif()

# Python bindings
if(NEML2_PYBIND)
      add_subdirectory(python)
endif()

# Documentation
if(NEML2_DOC)
      add_subdirectory(doc)
endif()
