# ----------------------------------------------------------------------------
# Project-level settings, options, and flags
# ----------------------------------------------------------------------------
option(NEML2_UNIT "Build NEML2 unit tests" ON)
option(NEML2_REGRESSION "Build NEML2 regression tests" ON)
option(NEML2_VERIFICATION "Build NEML2 verification tests" ON)
option(NEML2_BENCHMARK "Build NEML2 benchmark tests" OFF)
option(NEML2_PROFILER "Build NEML2 profiler" OFF)

# ----------------------------------------------------------------------------
# Dependencies and 3rd party packages
# ----------------------------------------------------------------------------
if(NEML2_UNIT OR NEML2_REGRESSION OR NEML2_VERIFICATION OR NEML2_BENCHMARK)
      FetchContent_MakeAvailable(Catch2)

      # Catch2 is built as a static library, so we MUST make sure the compile definitions are compatible with ours
      if(Torch_CXX11_ABI)
            target_compile_definitions(Catch2 PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
      else()
            target_compile_definitions(Catch2 PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
      endif()
endif()

# ----------------------------------------------------------------------------
# Subdirectories
# ----------------------------------------------------------------------------
# Test utilities
add_subdirectory(src)

# Unit tests
if(NEML2_UNIT)
      add_subdirectory(unit build/unit)
endif()

# Regression tests
if(NEML2_REGRESSION)
      add_subdirectory(regression build/regression)
endif()

# Verification tests
if(NEML2_VERIFICATION)
      add_subdirectory(verification build/verification)
endif()

# Benchmark tests
if(NEML2_BENCHMARK)
      add_subdirectory(benchmark build/benchmark)
endif()

# Profiler
if(NEML2_PROFILER)
      add_subdirectory(profiler build/profiler)
endif()

# pytests
if(NEML2_PYBIND)
      add_subdirectory(python build/python)
endif()
