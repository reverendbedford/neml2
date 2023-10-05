include(Macros)

# Add all the source files
file(GLOB_RECURSE SRCS *.cxx)
add_library(neml2 ${SRCS})

# compile options
target_compile_options(neml2 PUBLIC -Wall -Wextra -pedantic -Werror)

REGISTERUNITYGROUP(neml2 "NEML2" .)

# NEML2 headers
target_include_directories(neml2 PRIVATE "${PROJECT_SOURCE_DIR}/include")

# torch libraries
target_link_libraries(neml2 ${TORCH_LIBRARIES})

# wasphit for parsing
target_link_libraries(neml2 wasphit)

# Make scalar type configurable:
set(DTYPE_LIST UInt8 Int8 Int16 Int32 Int64 Float16 Float32 Float64)
set(DTYPE "Float64" CACHE STRING "default scalar type. Candidates are ${DTYPE_LIST}")
target_compile_definitions(neml2 PUBLIC DTYPE=${DTYPE})
message(STATUS "Configuring with default scalar type: ${DTYPE}")

# For testing
target_compile_definitions(neml2 PUBLIC WASP_ENABLED)
