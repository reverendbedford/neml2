include(NEML2UnityGroup)

# Catch2, for testing
add_subdirectory(${NEML2_SOURCE_DIR}/extern/Catch2 ${NEML2_BINARY_DIR}/extern/Catch2)
list(APPEND CMAKE_MODULE_PATH ${NEML2_SOURCE_DIR}/extern/Catch2/contrib)

# include(CTest)
include(Catch)

# Test utilities
add_subdirectory(src)

# ###################################################
# Unit tests
# ###################################################
option(NEML2_UNIT "Build NEML2 unit tests" ON)

if(NEML2_UNIT)
      file(GLOB_RECURSE UNIT_TESTS unit/*.cxx)
      add_executable(unit_tests
            ${TEST_UTILS}
            ${UNIT_TESTS}
      )

      target_compile_options(unit_tests PUBLIC -Wall -Wextra -pedantic -Werror)
      register_unity_group(unit_tests "Unit test" unit)
      target_link_libraries(unit_tests Catch2::Catch2)
      target_link_libraries(unit_tests neml2 testutils)
      target_include_directories(unit_tests PUBLIC ${NEML2_SOURCE_DIR}/tests/include)
      catch_discover_tests(unit_tests)

      if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
            add_custom_command(TARGET unit_tests
                  POST_BUILD
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${NEML2_SOURCE_DIR}/tests/unit ${NEML2_BINARY_DIR}/tests/unit
                  COMMENT "Creating symlink for unit tests"
            )
      endif()

      install(TARGETS unit_tests)
      install(DIRECTORY unit
            TYPE BIN
            FILES_MATCHING
            PATTERN "*.i"
            PATTERN "*.pt"
            PATTERN "*.txt"
            PATTERN "*.vtest"
            PATTERN "*.xml"
            PATTERN "*.py"
            PATTERN "*.pyi"
      )
endif()

# ###################################################
# Regression tests
# ###################################################
option(NEML2_REGRESSION "Build NEML2 regression tests" ON)

if(NEML2_REGRESSION)
      file(GLOB_RECURSE REGRESSION_TESTS regression/*.cxx)
      add_executable(regression_tests
            ${TEST_UTILS}
            ${REGRESSION_TESTS}
      )

      target_compile_options(regression_tests PUBLIC -Wall -Wextra -pedantic -Werror)
      register_unity_group(regression_tests "Regression test" regression)
      target_link_libraries(regression_tests Catch2::Catch2)
      target_link_libraries(regression_tests neml2 testutils)
      target_include_directories(regression_tests PUBLIC "${NEML2_SOURCE_DIR}/tests/include")
      catch_discover_tests(regression_tests)

      if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
            add_custom_command(TARGET regression_tests
                  POST_BUILD
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${NEML2_SOURCE_DIR}/tests/regression ${NEML2_BINARY_DIR}/tests/regression
                  COMMENT "Creating symlink for regression tests"
            )
      endif()

      install(TARGETS regression_tests)
      install(DIRECTORY regression
            TYPE BIN
            FILES_MATCHING
            PATTERN "*.i"
            PATTERN "*.pt"
            PATTERN "*.txt"
            PATTERN "*.vtest"
            PATTERN "*.xml"
            PATTERN "*.py"
            PATTERN "*.pyi"
      )
endif()

# ###################################################
# Verification tests
# ###################################################
option(NEML2_VERIFICATION "Build NEML2 verification tests" ON)

if(NEML2_VERIFICATION)
      file(GLOB_RECURSE VERIFICATION_TESTS verification/*.cxx)
      add_executable(verification_tests
            ${TEST_UTILS}
            ${VERIFICATION_TESTS}
      )

      target_compile_options(verification_tests PUBLIC -Wall -Wextra -pedantic -Werror)
      register_unity_group(verification_tests "Verification test" verification)
      target_link_libraries(verification_tests Catch2::Catch2)
      target_link_libraries(verification_tests neml2 testutils)
      target_include_directories(verification_tests PUBLIC "${NEML2_SOURCE_DIR}/tests/include")
      catch_discover_tests(verification_tests)

      if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
            add_custom_command(TARGET verification_tests
                  POST_BUILD
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${NEML2_SOURCE_DIR}/tests/verification ${NEML2_BINARY_DIR}/tests/verification
                  COMMENT "Creating symlink for verification tests"
            )
      endif()

      install(TARGETS verification_tests)
      install(DIRECTORY verification
            TYPE BIN
            FILES_MATCHING
            PATTERN "*.i"
            PATTERN "*.pt"
            PATTERN "*.txt"
            PATTERN "*.vtest"
            PATTERN "*.xml"
            PATTERN "*.py"
            PATTERN "*.pyi"
      )
endif()

# ###################################################
# Benchmarks
# ###################################################
option(NEML2_BENCHMARK "Build NEML2 benchmark tests" OFF)

if(NEML2_BENCHMARK)
      file(GLOB_RECURSE BENCHMARK_TESTS benchmark/*.cxx)
      add_executable(benchmark_tests
            ${TEST_UTILS}
            ${BENCHMARK_TESTS}
      )

      # compile options
      target_compile_options(benchmark_tests PUBLIC -Wall -Wextra -pedantic -Werror)

      register_unity_group(benchmark_tests "Benchmark test" benchmark)
      target_link_libraries(benchmark_tests Catch2::Catch2)
      target_link_libraries(benchmark_tests neml2 testutils)
      target_include_directories(benchmark_tests PUBLIC "${NEML2_SOURCE_DIR}/tests/include")
      catch_discover_tests(benchmark_tests)

      if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
            add_custom_command(TARGET benchmark_tests
                  POST_BUILD
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${NEML2_SOURCE_DIR}/tests/benchmark ${NEML2_BINARY_DIR}/tests/benchmark
                  COMMENT "Creating symlink for benchmark tests"
            )
      endif()

      install(TARGETS benchmark_tests)
      install(DIRECTORY benchmark
            TYPE BIN
            FILES_MATCHING
            PATTERN "*.i"
            PATTERN "*.pt"
            PATTERN "*.txt"
            PATTERN "*.vtest"
            PATTERN "*.xml"
            PATTERN "*.py"
            PATTERN "*.pyi"
      )
endif()

# ###################################################
# Profiling
# ###################################################
option(NEML2_PROFILING "Build NEML2 profiling tests" OFF)

if(NEML2_PROFILING)
      # gperftools for profiling
      add_subdirectory(${NEML2_SOURCE_DIR}/extern/gperftools ${NEML2_BINARY_DIR}/extern/gperftools EXCLUDE_FROM_ALL)
      file(GLOB_RECURSE PROFILING_TESTS profiling/*.cxx)
      add_executable(profiling_tests
            ${TEST_UTILS}
            ${PROFILING_TESTS}
      )

      register_unity_group(profiling_tests "Profiling test" profiling)
      target_compile_options(profiling_tests PUBLIC -Wall -Wextra -pedantic -Werror)
      target_link_options(profiling_tests PRIVATE "-Wl,-no-as-needed")
      target_link_libraries(profiling_tests neml2 testutils profiler)
      target_include_directories(profiling_tests PUBLIC "${NEML2_SOURCE_DIR}/tests/include")

      if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
            add_custom_command(TARGET profiling_tests
                  POST_BUILD
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${NEML2_SOURCE_DIR}/tests/profiling ${NEML2_BINARY_DIR}/tests/profiling
                  COMMENT "Creating symlink for profiling tests"
            )
      endif()

      install(TARGETS profiling_tests)
      install(DIRECTORY profiling
            TYPE BIN
            FILES_MATCHING
            PATTERN "*.i"
            PATTERN "*.pt"
            PATTERN "*.txt"
            PATTERN "*.vtest"
            PATTERN "*.xml"
            PATTERN "*.py"
            PATTERN "*.pyi"
      )
endif()

# ###################################################
# Python binding tests
# ###################################################
if(NEML2_PYBIND)
      configure_file(${NEML2_SOURCE_DIR}/tests/python/conftest.py.in ${NEML2_SOURCE_DIR}/tests/python/conftest.py)

      if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
            add_custom_target(link_pybind_tests ALL
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${NEML2_SOURCE_DIR}/tests/python ${NEML2_BINARY_DIR}/tests/python
                  COMMENT "Creating symlink for python binding tests"
            )
      endif()

      install(DIRECTORY python
            TYPE BIN
            FILES_MATCHING
            PATTERN "*.py"
      )
endif()
