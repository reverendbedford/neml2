include(NEML2UnityGroup)

file(GLOB_RECURSE srcs *.cxx)
add_executable(benchmark ${srcs})

target_compile_options(benchmark PRIVATE -Wall -Wextra -pedantic -Werror)
register_unity_group(benchmark .)
target_link_libraries(benchmark PRIVATE testutils)
target_link_libraries(benchmark PRIVATE Catch2::Catch2WithMain)

if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
  file(CREATE_LINK ${NEML2_SOURCE_DIR}/tests/benchmark ${NEML2_BINARY_DIR}/tests/benchmark SYMBOLIC)
endif()

install(TARGETS benchmark)
install(DIRECTORY .
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
