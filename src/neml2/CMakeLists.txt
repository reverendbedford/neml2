include(NEML2UnityGroup)

# Add all the source files
file(GLOB_RECURSE SRCS *.cxx)
add_library(neml2 SHARED ${SRCS})

# Make scalar type configurable:
if(NOT NEML2_DTYPE)
  set(NEML2_DTYPE "Float64" CACHE STRING "Default NEML2 scalar type." FORCE)
  set_property(CACHE NEML2_DTYPE PROPERTY STRINGS "Float16" "Float32" "Float64")
endif()

message(STATUS "Configuring with default scalar type: ${NEML2_DTYPE}")

# Also want to configure an int type for specialized int tensors
if(NOT NEML2_INT_DTYPE)
  set(NEML2_INT_DTYPE "Int64" CACHE STRING "Default NEML2 integer scalar type." FORCE)
  set_property(CACHE NEML2_INT_DTYPE PROPERTY STRINGS "Int8" "Int16" "Int32" "Int64")
endif()

message(STATUS "Configuring with default integer scalar type: ${NEML2_INT_DTYPE}")

configure_file(
  ${NEML2_SOURCE_DIR}/include/neml2/base/config.h.in
  ${NEML2_SOURCE_DIR}/include/neml2/base/config.h
)

# NEML2 (private) compile options
target_compile_options(neml2 PRIVATE -Wall -Wextra -pedantic -Werror)

# Group source files together if UNITY build is requested
register_unity_group(neml2 "NEML2" .)

# NEML2 headers
file(GLOB_RECURSE _NEML2_HEADERS ${NEML2_SOURCE_DIR}/include/*.h)
target_include_directories(neml2 PUBLIC ${NEML2_SOURCE_DIR}/include ${NEML2_BINARY_DIR}/include)
target_sources(neml2
  PUBLIC
  FILE_SET HEADERS
  BASE_DIRS ${NEML2_SOURCE_DIR}/include
  FILES
  ${_NEML2_HEADERS}
)

# torch
target_include_directories(neml2 SYSTEM PUBLIC ${Torch_INCLUDE_DIRECTORIES})
target_link_directories(neml2 PUBLIC ${Torch_LINK_DIRECTORIES})
target_link_libraries(neml2 PUBLIC ${Torch_LIBRARIES})

if(Torch_CXX11_ABI)
  target_compile_definitions(neml2 PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
else()
  target_compile_definitions(neml2 PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
endif()

# hit for parsing
add_subdirectory("${NEML2_SOURCE_DIR}/extern/hit" "${NEML2_BINARY_DIR}/extern/hit")
target_link_libraries(neml2 PUBLIC hit)

install(
  TARGETS neml2
  FILE_SET HEADERS
  COMPONENT Development
)
