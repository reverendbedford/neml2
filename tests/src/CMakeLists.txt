file(GLOB_RECURSE TEST_UTILS *.cxx)
add_library(testutils SHARED ${TEST_UTILS})

target_link_libraries(testutils neml2)
target_include_directories(testutils PUBLIC "${NEML2_SOURCE_DIR}/tests/include")
target_include_directories(testutils PUBLIC "${NEML2_SOURCE_DIR}/include")

list(APPEND TEST_UTILS_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/neml2/lib)
list(APPEND TEST_UTILS_INSTALL_RPATH ${LIBTORCH_DIR}/lib)
set_target_properties(testutils PROPERTIES INSTALL_RPATH "${TEST_UTILS_INSTALL_RPATH}")
install(TARGETS testutils LIBRARY DESTINATION neml2/lib)
