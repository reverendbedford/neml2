include(NEML2UnityGroup)

# Add all the source files
file(GLOB_RECURSE SRCS *.cxx)
add_library(neml2 SHARED ${SRCS})

# NEML2 (private) compile options
target_compile_options(neml2 PRIVATE -Wall -Wextra -pedantic -Werror)

# Group source files together if UNITY build is requested
register_unity_group(neml2 "NEML2" .)

# NEML2 headers
target_include_directories(neml2 PUBLIC ${NEML2_SOURCE_DIR}/include ${NEML2_BINARY_DIR}/include)

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
target_link_libraries(neml2 PUBLIC hit)

# Make scalar type configurable:
if(NOT NEML2_DTYPE)
  set(NEML2_DTYPE "Float64" CACHE STRING "Default NEML2 scalar type." FORCE)
  set_property(CACHE NEML2_DTYPE PROPERTY STRINGS
    "UInt8" "Int8" "Int16" "Int32" "Int64" "Float16" "Float32" "Float64")
endif()

message(STATUS "Configuring with default scalar type: ${NEML2_DTYPE}")
target_compile_definitions(neml2 PUBLIC DTYPE=${NEML2_DTYPE})

# Also want to configure an int type for specialized int tensors
if(NOT NEML2_INT_DTYPE)
  set(NEML2_INT_DTYPE "Int64" CACHE STRING "Default NEML2 integer scalar type." FORCE)
  set_property(CACHE NEML2_INT_DTYPE PROPERTY STRINGS
    "Int8" "Int16" "Int32" "Int64")
endif()

message(STATUS "Configuring with default integer scalar type: ${NEML2_INT_DTYPE}")
configure_file(
  ${NEML2_SOURCE_DIR}/include/neml2/base/config.h.in
  ${NEML2_BINARY_DIR}/include/neml2/base/config.h
)
install(FILES ${NEML2_BINARY_DIR}/include/neml2/base/config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/neml2/include/base)

list(APPEND NEML2_INSTALL_RPATH lib)
list(APPEND NEML2_INSTALL_RPATH ${LIBTORCH_DIR}/lib)
set_target_properties(neml2 PROPERTIES INSTALL_RPATH "${NEML2_INSTALL_RPATH}")
install(TARGETS neml2 LIBRARY DESTINATION neml2/lib)

# Install public headers
# This is apparently the non-modern way as I am just copying all headers in the include directory
# This is because I am having trouble understanding the so-called "modern" way...
file(GLOB_RECURSE NEML2_PUBLIC_HEADERS ${NEML2_SOURCE_DIR}/include/*.h)

foreach(NEML2_PUBLIC_HEADER ${NEML2_PUBLIC_HEADERS})
  file(RELATIVE_PATH NEML2_PUBLIC_HEADER_RELPATH ${NEML2_SOURCE_DIR}/include/neml2 ${NEML2_PUBLIC_HEADER})
  cmake_path(GET NEML2_PUBLIC_HEADER_RELPATH PARENT_PATH NEML2_PUBLIC_HEADER_DIR)
  install(FILES ${NEML2_PUBLIC_HEADER} DESTINATION ${CMAKE_INSTALL_PREFIX}/neml2/include/${NEML2_PUBLIC_HEADER_DIR})
endforeach()
