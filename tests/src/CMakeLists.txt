file(GLOB_RECURSE TEST_UTILS *.cxx)
add_library(testutils OBJECT ${TEST_UTILS})
target_link_libraries(testutils PUBLIC neml2)
target_include_directories(testutils PUBLIC ${NEML2_SOURCE_DIR}/tests/include)
