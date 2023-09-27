include(Macros)

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
      file(GLOB_RECURSE UNIT_TESTS unit/*.cxx)
      add_executable(unit_tests
            ${TEST_UTILS}
            ${UNIT_TESTS}
      )
      REGISTERUNITYGROUP(unit_tests "Unit test" unit)
      target_link_libraries(unit_tests Catch2::Catch2)
      target_link_libraries(unit_tests neml2 testutils)
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
      file(GLOB_RECURSE REGRESSION_TESTS regression/*.cxx)
      add_executable(regression_tests
            ${TEST_UTILS}
            ${REGRESSION_TESTS}
      )
      REGISTERUNITYGROUP(regression_tests "Regression test" regression)
      target_link_libraries(regression_tests Catch2::Catch2)
      target_link_libraries(regression_tests neml2 testutils)
      target_include_directories(regression_tests PUBLIC
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/tests/include")
      catch_discover_tests(regression_tests)
endif()

# ###################################################
# Verification tests
# ###################################################
option(VERIFICATION "build verification tests" ON)

if(VERIFICATION)
      file(GLOB_RECURSE VERIFICATION_TESTS verification/*.cxx)
      add_executable(verification_tests
            ${TEST_UTILS}
            ${VERIFICATION_TESTS}
      )
      REGISTERUNITYGROUP(verification_tests "Verification test" verification)
      target_link_libraries(verification_tests Catch2::Catch2)
      target_link_libraries(verification_tests neml2 testutils)
      target_include_directories(verification_tests PUBLIC
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/tests/include")
      catch_discover_tests(verification_tests)
endif()

# ###################################################
# Benchmarks
# ###################################################
option(BENCHMARK "build benchmark tests" OFF)

if(BENCHMARK)
      file(GLOB_RECURSE BENCHMARK_TESTS benchmark/*.cxx)
      add_executable(benchmark_tests
            ${TEST_UTILS}
            ${BENCHMARK_TESTS}
      )
      REGISTERUNITYGROUP(benchmark_tests "Benchmark test" benchmark)
      target_link_libraries(benchmark_tests Catch2::Catch2)
      target_link_libraries(benchmark_tests neml2 testutils)
      target_include_directories(benchmark_tests PUBLIC
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/tests/include")
      catch_discover_tests(benchmark_tests)
endif()

# ###################################################
# Profiling
# ###################################################
option(PROFILING "build profiling tests" OFF)

if(PROFILING)
      if(DEFINED ENV{GPERFTOOLS_DIR})
            set(GPERFTOOLS_ROOT_DIR $ENV{GPERFTOOLS_DIR})
      else()
            message(FATAL_ERROR "Environment variable GPERFTOOLS_DIR must be set if profiling tests are turned on. GPERFTOOLS_DIR should point to the gperftools directory such that GPERFTOOLS_DIR/lib/libprofiler.* exist.")
      endif()

      file(GLOB_RECURSE PROFILING_TESTS profiling/*.cxx)
      add_executable(profiling_tests
            ${TEST_UTILS}
            ${PROFILING_TESTS}
      )
      REGISTERUNITYGROUP(profiling_tests "Profiling test" profiling)
      target_link_options(profiling_tests PUBLIC -Wl,--no-as-needed,-lprofiler,--as-needed)
      target_link_directories(profiling_tests PUBLIC ${GPERFTOOLS_ROOT_DIR}/lib)

      # target_link_libraries(profiling_tests profiler)
      target_link_libraries(profiling_tests neml2 testutils)
      target_include_directories(profiling_tests PUBLIC
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/tests/include")
endif()
