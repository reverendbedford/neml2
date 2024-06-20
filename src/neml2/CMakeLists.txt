include(NEML2UnityGroup)

# Add all the source files
file(GLOB_RECURSE SRCS *.cxx)
add_library(neml2 SHARED ${SRCS})
set_target_properties(neml2 PROPERTIES INSTALL_RPATH "${EXEC_DIR};${Torch_LINK_DIRECTORIES}")

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
register_unity_group(neml2 .)

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
install(TARGETS neml2 COMPONENT Development FILE_SET HEADERS)

# torch
target_include_directories(neml2 SYSTEM PUBLIC ${Torch_INCLUDE_DIRECTORIES})
target_link_directories(neml2 PUBLIC ${Torch_LINK_DIRECTORIES})
target_link_libraries(neml2 PUBLIC ${Torch_LIBRARIES})

# Do not track coverage info in WASP and HIT
set(CMAKE_BUILD_TYPE_CACHE ${CMAKE_BUILD_TYPE})
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)

# WASP for parsing
message(STATUS "Configuring WASP")
set(wasp_ENABLE_ALL_PACKAGES OFF CACHE INTERNAL "" FORCE)
set(wasp_ENABLE_wasphit ON CACHE INTERNAL "" FORCE)
set(wasp_ENABLE_testframework OFF CACHE INTERNAL "" FORCE)
set(wasp_ENABLE_TESTS OFF CACHE INTERNAL "" FORCE)
set(DISABLE_HIT_TYPE_PROMOTION ON CACHE INTERNAL "" FORCE)
FetchContent_MakeAvailable(wasp)
set_target_properties(waspcore wasphit
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  UNITY_BUILD OFF
)
target_sources(wasphit
  PUBLIC
  FILE_SET HEADERS
  BASE_DIRS ${wasp_SOURCE_DIR}
  FILES
  ${wasp_SOURCE_DIR}/wasphit/HITLexer.h
  ${wasp_SOURCE_DIR}/wasphit/HITParser.hpp
  ${wasp_SOURCE_DIR}/wasphit/HITInterpreter.h
  ${wasp_SOURCE_DIR}/wasphit/HITInterpreter.i.h
  ${wasp_SOURCE_DIR}/wasphit/HITNodeView.h
  ${wasp_SOURCE_DIR}/wasphit/HITNodeView.i.h
  ${wasp_SOURCE_DIR}/wasphit/HIT.lex
  ${wasp_SOURCE_DIR}/wasphit/HIT.bison
)
install(TARGETS wasphit COMPONENT Development FILE_SET HEADERS)

# HIT for parsing
message(STATUS "Configuring HIT")
FetchContent_MakeAvailable(hit)

add_library(hit SHARED
  ${hit_SOURCE_DIR}/src/hit/parse.cc
  ${hit_SOURCE_DIR}/src/hit/lex.cc
  ${hit_SOURCE_DIR}/src/hit/braceexpr.cc
)

set_target_properties(hit PROPERTIES UNITY_BUILD OFF)
target_include_directories(hit PUBLIC ${hit_SOURCE_DIR}/.. ${wasp_SOURCE_DIR})
target_link_libraries(hit PRIVATE wasphit)
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

# Recover the build type now that WASP and HIT has been added
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_CACHE} CACHE STRING "Choose the type of build." FORCE)
