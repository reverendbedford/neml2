# ----------------------------------------------------------------------------
# Dependencies and 3rd party packages
# ----------------------------------------------------------------------------
message(STATUS "Configuring Catch2")
FetchContent_MakeAvailable(Catch2)

# ----------------------------------------------------------------------------
# Subdirectories
# ----------------------------------------------------------------------------
# Test utilities
add_subdirectory(src)

# Unit tests
set(NEML2_UNIT ON CACHE BOOL "Build NEML2 unit tests")

if(NEML2_UNIT)
      add_subdirectory(unit)
endif()

# Regression tests
set(NEML2_REGRESSION ON CACHE BOOL "Build NEML2 regression tests")

if(NEML2_REGRESSION)
      add_subdirectory(regression)
endif()

# Verification tests
set(NEML2_VERIFICATION ON CACHE BOOL "Build NEML2 verification tests")

if(NEML2_VERIFICATION)
      add_subdirectory(verification)
endif()
