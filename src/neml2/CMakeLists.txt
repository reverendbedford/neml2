# Add all the source files
file(GLOB_RECURSE SRCS *.cxx)
add_library(batchedmat SHARED ${SRCS})

# My headers
target_include_directories(batchedmat PRIVATE "${CMAKE_SOURCE_DIR}/include")

# torch headers
# torch libraries
target_link_libraries(batchedmat ${TORCH_LIBRARIES})

# hit for parsing
target_link_libraries(batchedmat hit)
