# Register with CTest
include(CTest)
include(Catch)

# Test utilities
add_subdirectory(src)

# ###################################################
# Unit tests
# ###################################################
option(UNIT "build unit tests" ON)

if(UNIT)
      file(GLOB_RECURSE UNIT_TESTS unit/test_*.cxx)
      add_executable(unit_tests
            ${TEST_UTILS}
            ${UNIT_TESTS}
      )
      target_link_libraries(unit_tests Catch2::Catch2)
      target_link_libraries(unit_tests batchedmat testutils)
      target_include_directories(unit_tests PUBLIC
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/tests/include")
      catch_discover_tests(unit_tests)
endif()

# ###################################################
# Regression tests
# ###################################################
option(REGRESSION "build regression tests" ON)

if(REGRESSION)
      file(GLOB_RECURSE REGRESSION_TESTS regression/regression_*.cxx)
      add_executable(regression_tests
            ${TEST_UTILS}
            ${REGRESSION_TESTS}
      )
      target_link_libraries(regression_tests Catch2::Catch2)
      target_link_libraries(regression_tests batchedmat testutils)
      target_include_directories(regression_tests PUBLIC
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/tests/include")
      catch_discover_tests(regression_tests)
endif()

# ###################################################
# Profiling tests
# ###################################################
option(PROFILING "build profiling tests" OFF)

if(PROFILING)
      file(GLOB_RECURSE PROFILING_TESTS profiling/profile_*.cxx)
      add_executable(profiling_tests
            ${TEST_UTILS}
            ${PROFILING_TESTS}
      )
      target_link_libraries(profiling_tests Catch2::Catch2)
      target_link_libraries(profiling_tests batchedmat testutils)
      target_include_directories(profiling_tests PUBLIC
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/tests/include")
      catch_discover_tests(profiling_tests)
endif()
