include(NEML2UnityGroup)

file(GLOB_RECURSE srcs *.cxx)
add_executable(regression_tests ${srcs})
set_target_properties(regression_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NEML2_BINARY_DIR}/tests)

target_compile_options(regression_tests PRIVATE -Wall -Wextra -pedantic -Werror)
register_unity_group(regression_tests .)
target_link_libraries(regression_tests PRIVATE testutils)
target_link_libraries(regression_tests PRIVATE Catch2::Catch2WithMain)

if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
  file(CREATE_LINK ${NEML2_SOURCE_DIR}/tests/regression ${NEML2_BINARY_DIR}/tests/regression SYMBOLIC)
endif()

install(TARGETS regression_tests)
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
