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
set_property(GLOBAL PROPERTY JOB_POOLS CI=6) # Ninja only: limit number of jobs in a CI build
set(CMAKE_CXX_FLAGS_COVERAGE "-O0 -fprofile-arcs -ftest-coverage" CACHE STRING "Flags used by C++ compiler during coverage builds." FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(NEML2_WHEELS "Modify rpath to accomodate Python wheels" OFF)
option(NEML2_TESTS "Build NEML2 tests" ON)
option(NEML2_PYBIND "Build NEML2 Python binding" OFF)
option(NEML2_DOC "Build NEML2 documentation: doxygen" OFF)

# ----------------------------------------------------------------------------
# Dependencies and 3rd party packages
# ----------------------------------------------------------------------------
set(PYTORCH_VERSION "2.2.2")
set(DOXYGEN_VERSION "1.10.0")
set(DOXYGEN_AWESOME_VERSION "2.3.2")
set(PYBIND11_VERSION "2.12.0")
set(HIT_VERSION "100b575af08643b5e646cac8faff6c87dd1c15a7")
set(CATCH2_VERSION "3.5.4")
set(GPERFTOOLS_VERSION "2.15")
include(NEML2Dependencies)
find_package(Torch) # This gets redirected to our FindTorch.cmake

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
add_subdirectory(src/neml2) # base neml2 library

# tests
if(NEML2_TESTS)
      add_subdirectory(tests)
endif()

# Python bindings
if(NEML2_PYBIND)
      add_subdirectory(python)
endif()

# Documentation
if(NEML2_DOC)
      add_subdirectory(doc)
endif()
