include(NEML2UnityGroup)

# Add all the source files
file(GLOB_RECURSE SRCS *.cxx)
add_library(neml2 SHARED ${SRCS})
set_target_properties(neml2 PROPERTIES INSTALL_RPATH "${EXEC_DIR};${Torch_LINK_DIRECTORIES}")

configure_file(
  ${NEML2_SOURCE_DIR}/include/neml2/base/config.h.in
  ${NEML2_SOURCE_DIR}/include/neml2/base/config.h
)

# NEML2 (private) compile options
target_compile_options(neml2 PRIVATE -Wall -Wextra -pedantic -Werror)

# Group source files together if UNITY build is requested
register_unity_group(neml2 .)

# NEML2 headers
file(GLOB_RECURSE _NEML2_HEADERS ${NEML2_SOURCE_DIR}/include/*.h)
target_include_directories(neml2 PUBLIC ${NEML2_SOURCE_DIR}/include ${NEML2_SOURCE_DIR}/contrib)
target_link_libraries(neml2 PUBLIC Threads::Threads)
target_sources(neml2
  PUBLIC
  FILE_SET HEADERS
  BASE_DIRS ${NEML2_SOURCE_DIR}/include
  FILES
  ${_NEML2_HEADERS}
)
install(TARGETS neml2 COMPONENT Development FILE_SET HEADERS)
install(DIRECTORY ${NEML2_SOURCE_DIR}/contrib/csvparser TYPE INCLUDE)

# torch
target_include_directories(neml2 SYSTEM PUBLIC ${Torch_INCLUDE_DIRECTORIES})
target_link_directories(neml2 PUBLIC ${Torch_LINK_DIRECTORIES})
target_link_libraries(neml2 PUBLIC ${Torch_LIBRARIES})

# HIT (WASP) for parsing
message(STATUS "Configuring HIT")
FetchContent_MakeAvailable(hit)

add_library(hit SHARED
  ${hit_SOURCE_DIR}/src/hit/parse.cc
  ${hit_SOURCE_DIR}/src/hit/lex.cc
  ${hit_SOURCE_DIR}/src/hit/braceexpr.cc
)
add_dependencies(hit wasp)
set(wasp_INSTALL_DIR ${NEML2_BINARY_DIR}/wasp/install)
install(DIRECTORY
  ${wasp_INSTALL_DIR}/include/waspcore
  ${wasp_INSTALL_DIR}/include/wasphit
  TYPE INCLUDE
)

set_target_properties(hit PROPERTIES UNITY_BUILD OFF)
target_include_directories(hit PUBLIC ${hit_SOURCE_DIR}/.. ${wasp_INSTALL_DIR}/include)
target_link_libraries(hit PRIVATE waspcore wasphit)
target_link_directories(hit PRIVATE ${wasp_INSTALL_DIR}/lib)
target_sources(hit
  PUBLIC
  FILE_SET HEADERS
  BASE_DIRS ${hit_SOURCE_DIR}/include
  FILES
  ${hit_SOURCE_DIR}/include/hit/braceexpr.h
  ${hit_SOURCE_DIR}/include/hit/hit.h
  ${hit_SOURCE_DIR}/include/hit/lex.h
  ${hit_SOURCE_DIR}/include/hit/parse.h
)
install(TARGETS hit COMPONENT Development FILE_SET HEADERS)
target_link_libraries(neml2 PUBLIC hit)
