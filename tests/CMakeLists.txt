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
      SUBDIRLIST(UNIT_SUBDIRS unit)
      get_filename_component(UNIT_SRCROOT unit ABSOLUTE)

      FOREACH(subdir ${UNIT_SUBDIRS})
            SETUNITYGROUP("Unit test" ${UNIT_SRCROOT} ${subdir})
      ENDFOREACH()

      file(GLOB_RECURSE UNIT_TESTS unit/*.cxx)
      add_executable(unit_tests
            ${TEST_UTILS}
            ${UNIT_TESTS}
      )
      set_target_properties(unit_tests PROPERTIES UNITY_BUILD_MODE GROUP)
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
      SUBDIRLIST(REGRESSION_SUBDIRS regression)
      get_filename_component(REGRESSION_SRCROOT regression ABSOLUTE)

      FOREACH(subdir ${REGRESSION_SUBDIRS})
            SETUNITYGROUP("Regression test" ${REGRESSION_SRCROOT} ${subdir})
      ENDFOREACH()

      file(GLOB_RECURSE REGRESSION_TESTS regression/*.cxx)
      add_executable(regression_tests
            ${TEST_UTILS}
            ${REGRESSION_TESTS}
      )
      set_target_properties(regression_tests PROPERTIES UNITY_BUILD_MODE GROUP)
      target_link_libraries(regression_tests Catch2::Catch2)
      target_link_libraries(regression_tests neml2 testutils)
      target_include_directories(regression_tests PUBLIC
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/tests/include")
      catch_discover_tests(regression_tests)
endif()

# ###################################################
# Benchmarks
# ###################################################
option(BENCHMARK "build benchmark tests" OFF)

if(BENCHMARK)
      SUBDIRLIST(BENCHMARK_SUBDIRS benchmark)
      get_filename_component(BENCHMARK_SRCROOT benchmark ABSOLUTE)

      FOREACH(subdir ${BENCHMARK_SUBDIRS})
            SETUNITYGROUP("Benchmark test" ${BENCHMARK_SRCROOT} ${subdir})
      ENDFOREACH()

      file(GLOB_RECURSE BENCHMARK_TESTS benchmark/benchmark_*.cxx)
      add_executable(benchmark_tests
            ${TEST_UTILS}
            ${BENCHMARK_TESTS}
      )
      set_target_properties(benchmark_tests PROPERTIES UNITY_BUILD_MODE GROUP)
      target_link_libraries(benchmark_tests Catch2::Catch2)
      target_link_libraries(benchmark_tests neml2 testutils)
      target_include_directories(benchmark_tests PUBLIC
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/tests/include")
      catch_discover_tests(benchmark_tests)
endif()

# ###################################################
# Verification tests
# ###################################################
option(VERIFICATION "build verification tests" ON)

if(VERIFICATION)
      SUBDIRLIST(VERIFICATION_SUBDIRS verification)
      get_filename_component(VERIFICATION_SRCROOT verification ABSOLUTE)

      FOREACH(subdir ${VERIFICATION_SUBDIRS})
            SETUNITYGROUP("Verification test" ${VERIFICATION_SRCROOT} ${subdir})
      ENDFOREACH()

      file(GLOB_RECURSE VERIFICATION_TESTS verification/*.cxx)
      add_executable(verification_tests
            ${TEST_UTILS}
            ${VERIFICATION_TESTS}
      )
      set_target_properties(verification_tests PROPERTIES UNITY_BUILD_MODE GROUP)
      target_link_libraries(verification_tests Catch2::Catch2)
      target_link_libraries(verification_tests neml2 testutils)
      target_include_directories(verification_tests PUBLIC
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/tests/include")
      catch_discover_tests(verification_tests)
endif()
