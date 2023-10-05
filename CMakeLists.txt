CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

project(NEML2 LANGUAGES CXX)

# A trick to control submodule configuration verbosity
function(message)
      if(NOT MESSAGE_QUIET)
            _message(${ARGN})
      endif()
endfunction()

# ## Setup modules ###
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

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

# Add the unity option to the cache
option(CMAKE_UNITY_BUILD "Use a unity build" OFF)

# Build shared library by default
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# Enable testing
enable_testing()

# c++ 17 support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Location of current CPU torch
include(FetchContent)
FetchContent_Declare(
      torch
      URL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
)

# Select between system torch and the current (CPU) torch
if(DEFINED LIBTORCH_DIR)
      find_package(Torch REQUIRED HINTS "${LIBTORCH_DIR}/share/cmake")
else()
      FetchContent_MakeAvailable(torch)
      find_package(Torch REQUIRED HINTS "${torch_SOURCE_DIR}/share/cmake")
endif()

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
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/Catch2" "${CMAKE_BINARY_DIR}/extern/Catch2")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/extern/Catch2/contrib")

# WASP for parsing
option(wasp_ENABLE_ALL_PACKAGES "Enable WASP packages" OFF)
option(wasp_ENABLE_ALL_OPTIONAL_PACKAGES "Enable WASP optional packages" OFF)
option(wasp_ENABLE_TESTS "Enable WASP tests" OFF)
option(wasp_ENABLE_testframework "Enable WASP tests" OFF)
option(wasp_ENABLE_TEUCHOS_TIME_MONITOR "Enable support for Teuchos Time Monitors in all Trilinos packages that support it." OFF)
option(wasp_INSTALL_EXECUTABLES "Enable the installation of executables provided by the wasp packages." OFF)
option(wasp_ENABLE_wasphit "Enable the package wasphit." ON)
option(DISABLE_HIT_TYPE_PROMOTION "Disable HIT type-promotion" ON)

message(STATUS "Configuring WASP, this may take a few minutes...")
set(MESSAGE_QUIET ON)
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/wasp" "${CMAKE_BINARY_DIR}/extern/wasp" EXCLUDE_FROM_ALL)
unset(MESSAGE_QUIET)

# wasphit headers
target_include_directories(wasphit PUBLIC "${PROJECT_SOURCE_DIR}/extern/wasp")

# wasphit doesn't check signed comparison, unfortunately
target_compile_options(waspcore PRIVATE "-Wno-sign-compare")

# base library
add_subdirectory(src/neml2)

# tests
add_subdirectory(tests)

# Doxygen
option(DOCUMENTATION "Build documentation: doxygen" OFF)

if(DOCUMENTATION)
      add_subdirectory(doc)
endif()
