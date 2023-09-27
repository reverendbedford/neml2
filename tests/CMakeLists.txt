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

      # compile options
      target_compile_options(unit_tests PUBLIC -Wall -Wextra -pedantic -Werror)

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

      # compile options
      target_compile_options(regression_tests PUBLIC -Wall -Wextra -pedantic -Werror)

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

      # compile options
      target_compile_options(verification_tests PUBLIC -Wall -Wextra -pedantic -Werror)

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

      # compile options
      target_compile_options(benchmark_tests PUBLIC -Wall -Wextra -pedantic -Werror)

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
      # gperftools for profiling
      SET(BUILD_TESTING OFF CACHE BOOL "Build gperftools tests" FORCE)
      add_subdirectory(${PROJECT_SOURCE_DIR}/extern/gperftools ${PROJECT_SOURCE_DIR}/extern/gperftools)

      set_target_properties(tcmalloc_internal_object PROPERTIES UNITY_BUILD OFF)
      set_target_properties(tcmalloc_and_profiler PROPERTIES UNITY_BUILD OFF)
      set_target_properties(tcmalloc_and_profiler_static PROPERTIES UNITY_BUILD OFF)

      file(GLOB_RECURSE PROFILING_TESTS profiling/*.cxx)
      add_executable(profiling_tests
            ${TEST_UTILS}
            ${PROFILING_TESTS}
      )
      REGISTERUNITYGROUP(profiling_tests "Profiling test" profiling)

      # compile options
      target_compile_options(profiling_tests PUBLIC -Wall -Wextra -pedantic -Werror)
      target_link_options(profiling_tests PRIVATE "-Wl,-no-as-needed")
      target_link_libraries(profiling_tests neml2 testutils profiler)
      target_include_directories(profiling_tests PUBLIC
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/tests/include")
endif()
