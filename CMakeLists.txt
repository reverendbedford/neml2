CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

project(BATCHEDMAT LANGUAGES CXX)

# Let's be very careful
add_compile_options(-Wall -Wextra -pedantic -Werror)

# For debugging
option(DEBUG "include debugging information" OFF)

if(DEBUG)
  message(STATUS "Configure to include debugging information")
  add_compile_options(-g -O0)
endif()

# For coverage
option(COVERAGE "include coverage information" OFF)

if(COVERAGE)
  message(STATUS "Configure to include coverage information")
  add_compile_options(-O0 -fprofile-arcs -ftest-coverage)
  add_link_options(--coverage)
endif()

# Enable testing
enable_testing()

# c++ 17 support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# torch, for which we are going to rely on the system for now
find_package(Torch REQUIRED)

# Catch2, for testing
find_package(Catch2 REQUIRED)

# MKL
message(STATUS "MKL_ROOT=${MKL_ROOT}")
LINK_DIRECTORIES(${MKL_ROOT}/lib)

# base library
add_subdirectory(src)

# tests
add_subdirectory(tests)

# Doxygen
option(DOCUMENTATION "Build documentation: doxygen" OFF)

if(DOCUMENTATION)
  add_subdirectory(doc)
endif()
