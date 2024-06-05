# ----------------------------------------------------------------------------
# Project-level settings, options, and flags
# ----------------------------------------------------------------------------
option(NEML2_UNIT "Build NEML2 unit tests" ON)
option(NEML2_REGRESSION "Build NEML2 regression tests" ON)
option(NEML2_VERIFICATION "Build NEML2 verification tests" ON)

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
if(NEML2_UNIT)
      add_subdirectory(unit)
endif()

# Regression tests
if(NEML2_REGRESSION)
      add_subdirectory(regression)
endif()

# Verification tests
if(NEML2_VERIFICATION)
      add_subdirectory(verification)
endif()
