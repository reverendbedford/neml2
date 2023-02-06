# Add all the source files
file(GLOB_RECURSE SRCS *.cxx)
add_library(neml2 SHARED ${SRCS})

# My headers
target_include_directories(neml2 PRIVATE "${CMAKE_SOURCE_DIR}/include")

# torch headers
# torch libraries
target_link_libraries(neml2 ${TORCH_LIBRARIES})

# hit for parsing
target_link_libraries(neml2 hit)
