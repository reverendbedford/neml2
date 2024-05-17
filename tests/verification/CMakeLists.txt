include(NEML2UnityGroup)

file(GLOB_RECURSE srcs *.cxx)
add_executable(verification_tests ${srcs})
set_target_properties(verification_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NEML2_BINARY_DIR}/tests)

target_compile_options(verification_tests PRIVATE -Wall -Wextra -pedantic -Werror)
register_unity_group(verification_tests .)
target_link_libraries(verification_tests PRIVATE testutils)
target_link_libraries(verification_tests PRIVATE Catch2::Catch2WithMain)

if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
  file(CREATE_LINK ${NEML2_SOURCE_DIR}/tests/verification ${NEML2_BINARY_DIR}/tests/verification SYMBOLIC)
endif()

install(TARGETS verification_tests)
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
