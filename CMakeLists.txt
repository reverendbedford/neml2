CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

project(BATCHEDMAT LANGUAGES CXX)

# ## Setup modules ###
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Werror, etc
add_compile_options(-Wall -Wextra -pedantic -Werror)

# This is aggravating but occurs because pre-packaged torch uses
# _GLIBCXX_USE_CXX11_ABI=0.  If you roll your own torch this may
# not occur, so this flag could be dangerous in the future
option(CXX11ABI "Whether to let GLIBCXX use cxx11 ABI" ON)

if(NOT CXX11ABI)
      add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

# Accept Release, Debug, and RelWithDebInfo, add Coverage build types
set(CMAKE_CXX_FLAGS_COVERAGE
      "-O0 -fprofile-arcs -ftest-coverage"
      CACHE STRING "Flags used by C++ compiler during coverage builds."
      FORCE)

if(NOT CMAKE_BUILD_TYPE)
      SET(CMAKE_BUILD_TYPE Debug
            CACHE STRING "Choose the type of build : None Debug Release RelWithDebInfo MinSizeRel Coverage."
            FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Enable testing
enable_testing()

# c++ 17 support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# torch, for which we are going to rely on the system for now
find_package(Torch REQUIRED)

# Catch2, for testing
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/Catch2")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/extern/Catch2/contrib")

# hit for parsing
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/hit")

# MKL (I don't think this should be here)
message(STATUS "MKL_ROOT=${MKL_ROOT}")
LINK_DIRECTORIES(${MKL_ROOT}/lib)

# base library
add_subdirectory(src/neml2)

# tests
add_subdirectory(tests)

# Doxygen
option(DOCUMENTATION "Build documentation: doxygen" OFF)

if(DOCUMENTATION)
      add_subdirectory(doc)
endif()
