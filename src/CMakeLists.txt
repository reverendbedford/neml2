# Add all the source files
add_library(batchedmat
      BatchedScalar.cxx
      BatchedSymR2.cxx
      BatchedSymSymR4.cxx
      Scalar.cxx
      SymR2.cxx
      SymSymR4.cxx
      StateBase.cxx
      State.cxx
      StateDerivative.cxx
      StateInfo.cxx
      )

# My headers
target_include_directories(batchedmat PRIVATE "../include")

# torch headers
# torch libraries
target_link_libraries(batchedmat ${TORCH_LIBRARIES})
