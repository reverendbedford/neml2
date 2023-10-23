CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

project(NEML2 LANGUAGES CXX)

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

# Enable testing
enable_testing()

# c++ 17 support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress the warning related to the new policy on fetch content's timestamp
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
      cmake_policy(SET CMP0135 NEW)
endif()

# Select between system torch and the current (CPU) torch
if(NOT DEFINED LIBTORCH_DIR)
      # Location of current CPU torch
      include(FetchContent)
      if (UNIX)
            if (NOT APPLE)
                  FetchContent_Declare(torch URL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip)
            else()
                  if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
                        set(APPLE_SILICON ON)
                        # This URL will need to be maintained
                        FetchContent_Declare(torch URL https://files.pythonhosted.org/packages/7b/7c/4d8728e6f8dbe2b8af054bd92c290d94c633270443514e3ee4b768125cf9/torch-2.1.0-cp311-none-macosx_11_0_arm64.whl)
                  else()
                        set(APPLE_SILICON OFF)
                        FetchContent_Declare(torch URL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip)
                  endif()
                  message(WARNING ${CMAKE_SYSTEM_PROCESSOR})
            endif()
      else()
            message(FATAL_ERROR "We only download a default libtorch on linux and macos. For other operating systems, please specify LIBTORCH_DIR.")
      endif()
      set(FETCHCONTENT_QUIET FALSE)
      FetchContent_MakeAvailable(torch)
      set(FETCHCONTENT_QUIET TRUE)
      if (APPLE AND APPLE_SILICON)
            set(LIBTORCH_DIR ${torch_SOURCE_DIR}/torch)
      else()
            set(LIBTORCH_DIR ${torch_SOURCE_DIR})
      endif()
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
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/Catch2" "${CMAKE_BINARY_DIR}/extern/Catch2")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/extern/Catch2/contrib")

# hit for parsing
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/hit" "${CMAKE_BINARY_DIR}/extern/hit")

# base library
add_subdirectory(src/neml2)

# tests
add_subdirectory(tests)

# Doxygen
option(DOCUMENTATION "Build documentation: doxygen" OFF)

if(DOCUMENTATION)
      add_subdirectory(doc)
endif()
