# HIT for model (de)serialization
add_library(hit
  ${CMAKE_SOURCE_DIR}/extern/hit/parse.cc
  ${CMAKE_SOURCE_DIR}/extern/hit/lex.cc
  ${CMAKE_SOURCE_DIR}/extern/hit/braceexpr.cc
)
target_include_directories(hit PUBLIC ${CMAKE_SOURCE_DIR}/extern/hit)

# Add all the source files
file(GLOB_RECURSE SRCS *.cxx)
add_library(batchedmat ${SRCS})

# My headers
target_include_directories(batchedmat PRIVATE "../include")

# torch headers
# torch libraries
target_link_libraries(batchedmat ${TORCH_LIBRARIES})
target_link_libraries(batchedmat hit)
