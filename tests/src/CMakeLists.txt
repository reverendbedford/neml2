file(GLOB_RECURSE TEST_UTILS *.cxx)
add_library(testutils SHARED ${TEST_UTILS})

target_link_libraries(testutils neml2)
target_include_directories(testutils PUBLIC "${NEML2_SOURCE_DIR}/tests/include")
target_include_directories(testutils PUBLIC "${NEML2_SOURCE_DIR}/include")
