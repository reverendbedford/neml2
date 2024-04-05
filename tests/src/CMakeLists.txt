file(GLOB_RECURSE TEST_UTILS *.cxx)
add_library(testutils SHARED ${TEST_UTILS})
target_link_libraries(testutils PUBLIC neml2)
target_include_directories(testutils PUBLIC ${NEML2_SOURCE_DIR}/tests/include)
set_target_properties(testutils
  PROPERTIES
  INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${Torch_LINK_DIRECTORIES}"
)
install(TARGETS testutils)
