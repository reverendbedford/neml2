include(NEML2UnityGroup)

if(NEML2_UNIT OR NEML2_REGRESSION OR NEML2_VERIFICATION OR NEML2_BENCHMARK)
      # Catch2, for testing
      FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG b5373dadca40b7edc8570cf9470b9b1cb1934d40 # version 3.5.4
      )
      message(STATUS "Downloading Catch2, this may take a few minutes.")
      FetchContent_MakeAvailable(Catch2)

      # Catch2 v3 is built as a static library, so we MUST make sure the compile definitions are compatible with ours
      if(Torch_CXX11_ABI)
            target_compile_definitions(Catch2 PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
      else()
            target_compile_definitions(Catch2 PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
      endif()
endif()

# Test utilities
add_subdirectory(src)

# ###################################################
# Unit tests
# ###################################################
option(NEML2_UNIT "Build NEML2 unit tests" ON)

if(NEML2_UNIT)
      file(GLOB_RECURSE UNIT_TESTS unit/*.cxx)
      add_executable(unit_tests ${UNIT_TESTS})

      target_compile_options(unit_tests PRIVATE -Wall -Wextra -pedantic -Werror)
      register_unity_group(unit_tests "Unit test" unit)
      target_link_libraries(unit_tests PRIVATE testutils)
      target_link_libraries(unit_tests PRIVATE Catch2::Catch2WithMain)

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
      add_executable(regression_tests ${REGRESSION_TESTS})

      target_compile_options(regression_tests PRIVATE -Wall -Wextra -pedantic -Werror)
      register_unity_group(regression_tests "Regression test" regression)
      target_link_libraries(regression_tests PRIVATE testutils)
      target_link_libraries(regression_tests PRIVATE Catch2::Catch2WithMain)

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
      add_executable(verification_tests ${VERIFICATION_TESTS})

      target_compile_options(verification_tests PRIVATE -Wall -Wextra -pedantic -Werror)
      register_unity_group(verification_tests "Verification test" verification)
      target_link_libraries(verification_tests PRIVATE testutils)
      target_link_libraries(verification_tests PRIVATE Catch2::Catch2WithMain)

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
      add_executable(benchmark_tests ${BENCHMARK_TESTS})

      # compile options
      target_compile_options(benchmark_tests PRIVATE -Wall -Wextra -pedantic -Werror)

      register_unity_group(benchmark_tests "Benchmark test" benchmark)
      target_link_libraries(benchmark_tests PRIVATE testutils)
      target_link_libraries(benchmark_tests PRIVATE Catch2::Catch2WithMain)

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
      # gperftools
      FetchContent_Declare(
            gperftools
            GIT_REPOSITORY https://github.com/gperftools/gperftools.git
            GIT_TAG gperftools-2.15
            EXCLUDE_FROM_ALL
      )
      message(STATUS "Downloading gperftools, this may take a few minutes.")
      FetchContent_MakeAvailable(gperftools)

      file(GLOB_RECURSE PROFILING_TESTS profiling/*.cxx)
      add_executable(profiling_tests ${PROFILING_TESTS})

      register_unity_group(profiling_tests "Profiling test" profiling)
      target_compile_options(profiling_tests PRIVATE -Wall -Wextra -pedantic -Werror)
      target_link_options(profiling_tests PRIVATE "-Wl,-no-as-needed")
      target_link_libraries(profiling_tests PRIVATE testutils profiler)

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
