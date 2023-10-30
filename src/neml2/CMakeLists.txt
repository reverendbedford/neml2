include(NEML2UnityGroup)

# Add all the source files
file(GLOB_RECURSE SRCS *.cxx)
add_library(neml2 SHARED ${SRCS})

# compile options
target_compile_options(neml2 PUBLIC -Wall -Wextra -pedantic -Werror)

register_unity_group(neml2 "NEML2" .)

# My headers
target_include_directories(neml2 PUBLIC ${NEML2_SOURCE_DIR}/include ${NEML2_BINARY_DIR}/include)

# torch headers
# torch libraries
target_link_libraries(neml2 ${TORCH_LIBRARIES})

# hit for parsing
target_link_libraries(neml2 hit)

# Make scalar type configurable:
if(NOT NEML2_DTYPE)
  set(NEML2_DTYPE "Float64" CACHE STRING "Default NEML2 scalar type." FORCE)
  set_property(CACHE NEML2_DTYPE PROPERTY STRINGS
    "UInt8" "Int8" "Int16" "Int32" "Int64" "Float16" "Float32" "Float64")
endif()

message(STATUS "Configuring with default scalar type: ${NEML2_DTYPE}")

# Also want to configure an int type for specialized int tensors
if(NOT NEML2_INT_DTYPE)
  set(NEML2_INT_DTYPE "Int64" CACHE STRING "Default NEML2 integer scalar type." FORCE)
  set_property(CACHE NEML2_INT_DTYPE PROPERTY STRINGS
    "Int8" "Int16" "Int32" "Int64")
endif()

message(STATUS "Configuring with default integer scalar type: ${NEML2_INT_DTYPE}")
configure_file(${NEML2_SOURCE_DIR}/include/neml2/base/config.h.in
  ${NEML2_BINARY_DIR}/include/neml2/base/config.h)
