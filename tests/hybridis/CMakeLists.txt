file(GLOB_RECURSE srcs *.cxx)
add_executable(hybridis_tests ${srcs})
set_target_properties(hybridis_tests PROPERTIES INSTALL_RPATH "${EXEC_DIR}/../lib;${Torch_LINK_DIRECTORIES}")

target_compile_options(hybridis_tests PRIVATE -Wall -Wextra -pedantic -Werror)
target_link_libraries(hybridis_tests PUBLIC hybridis Catch2::Catch2WithMain)

install(TARGETS hybridis_tests COMPONENT Development)
install(DIRECTORY .
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/hybridis
  COMPONENT Development
  FILES_MATCHING
  PATTERN "*.i"
  PATTERN "*.pt"
  PATTERN "*.vtest"
  PATTERN "*.xml"
)
