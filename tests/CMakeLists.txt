# Should we do these one at a time or all at once?
add_executable(unit_tests 
      test.cxx
      test_BatchTensor.cxx
      test_FixedDimTensor.cxx)

# Libraries and headers
target_link_libraries(unit_tests Catch2::Catch2)
target_link_libraries(unit_tests batchedmat)
target_include_directories(unit_tests PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Register with CTest
include(CTest)
include(Catch)
catch_discover_tests(unit_tests)
