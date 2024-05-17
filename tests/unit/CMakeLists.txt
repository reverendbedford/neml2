include(NEML2UnityGroup)

file(GLOB_RECURSE srcs *.cxx)
add_executable(unit_tests ${srcs})
set_target_properties(unit_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NEML2_BINARY_DIR}/tests)

target_compile_options(unit_tests PRIVATE -Wall -Wextra -pedantic -Werror)
register_unity_group(unit_tests .)
target_link_libraries(unit_tests PRIVATE testutils)
target_link_libraries(unit_tests PRIVATE Catch2::Catch2WithMain)

if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
  file(CREATE_LINK ${NEML2_SOURCE_DIR}/tests/unit ${NEML2_BINARY_DIR}/tests/unit SYMBOLIC)
endif()

install(TARGETS unit_tests)
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
