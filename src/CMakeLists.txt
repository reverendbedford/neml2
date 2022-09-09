# Add all the source files
add_library(batchedmat
      BatchedScalar.cxx
      BatchedSymR2.cxx
      BatchedSymSymR4.cxx
      Scalar.cxx
      SymR2.cxx
      SymSymR4.cxx
      )

# My headers
target_include_directories(batchedmat PRIVATE "../include")

# torch headers
# torch libraries
target_link_libraries(batchedmat ${TORCH_LIBRARIES})

# Just a dumb test
add_executable(test test.cxx)
target_link_libraries(test ${TORCH_LIBRARIES} batchedmat)
target_include_directories(test PUBLIC "../include")
