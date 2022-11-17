# Should we do these one at a time or all at once?
file(GLOB_RECURSE TEST_UTILS src/*.cxx)
file(GLOB_RECURSE UNIT_TESTS unit/test_*.cxx)
file(GLOB_RECURSE REGRESSION_TESTS regression/regression_*.cxx)
file(GLOB_RECURSE PROFILING_TESTS profiling/profile_*.cxx)
add_executable(tests
      ${TEST_UTILS}
      ${UNIT_TESTS}
      ${REGRESSION_TESTS}
      ${PROFILING_TESTS}
)

# Libraries and headers
target_link_libraries(tests Catch2::Catch2)
target_link_libraries(tests batchedmat)
target_include_directories(tests PUBLIC 
      "${CMAKE_SOURCE_DIR}/include" 
      "${CMAKE_SOURCE_DIR}/tests/include")

# For benchmarking
option(BENCHMARK "include benchmark tests" OFF)

if(BENCHMARK)
      target_compile_definitions(unit_tests PUBLIC ENABLE_BENCHMARK)
endif()

# Register with CTest
include(CTest)
include(Catch)

catch_discover_tests(tests)
