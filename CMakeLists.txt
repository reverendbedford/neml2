cmake_minimum_required(VERSION 3.21)

project(NEML2 LANGUAGES CXX)

# Setup modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${NEML2_SOURCE_DIR}/cmake/Modules/")

# Ninja only: limit number of jobs in a CI build
set_property(GLOBAL PROPERTY JOB_POOLS CI=6)

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

# Accept Release, Debug, and RelWithDebInfo, add Coverage build types
set(CMAKE_CXX_FLAGS_COVERAGE
      "-O0 -fprofile-arcs -ftest-coverage"
      CACHE STRING "Flags used by C++ compiler during coverage builds."
      FORCE)

# Set the default build type
if(NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE "Debug" CACHE
            STRING "Choose the type of build." FORCE)

      # Set the possible values of build type for cmake-gui
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "Coverage")
endif()

# Add the unity option to the cache
option(CMAKE_UNITY_BUILD "Use a unity build" OFF)

# c++ 17 support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch)

# Install rpath, important for a relocatable install
if(UNIX)
      if(APPLE)
            set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/lib;${Torch_LINK_DIRECTORIES}")
      else()
            set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/lib;${Torch_LINK_DIRECTORIES}")
      endif()
endif()

# base library
add_subdirectory(src/neml2)

# testing
option(NEML2_TESTS "Build NEML2 tests" ON)

if(NEML2_TESTS)
      add_subdirectory(tests)
endif()

# Doxygen
option(NEML2_DOC "Build NEML2 documentation: doxygen" OFF)

if(NEML2_DOC)
      add_subdirectory(doc)
endif()

# Python binding
option(NEML2_PYBIND "Build NEML2 Python binding" OFF)

if(NEML2_PYBIND)
      add_subdirectory(python)
endif()
