include(NEML2UnityGroup)

FetchContent_Populate(gperftools)
add_subdirectory(${gperftools_SOURCE_DIR} ${gperftools_BINARY_DIR} EXCLUDE_FROM_ALL)

file(GLOB_RECURSE srcs *.cxx)
add_executable(profiler ${srcs})
register_unity_group(profiler .)
target_compile_options(profiler PRIVATE -Wall -Wextra -pedantic -Werror)
target_link_options(profiler PRIVATE "-Wl,-no-as-needed")
target_link_libraries(profiler PRIVATE testutils profiler)

if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
  file(CREATE_LINK ${NEML2_SOURCE_DIR}/tests/profiler ${NEML2_BINARY_DIR}/tests/profiler SYMBOLIC)
endif()

install(TARGETS profiler)
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
