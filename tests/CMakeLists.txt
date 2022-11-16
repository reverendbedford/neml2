# Should we do these one at a time or all at once?
add_executable(unit_tests
      test.cxx
      test_BatchTensor.cxx
      test_FixedDimTensor.cxx
      test_LabeledTensor.cxx
      test_StateInfo.cxx
      test_State.cxx
      test_StateDerivative.cxx
      test_Scalar.cxx
      test_SymR2.cxx
      test_SymSymR4.cxx
      test_SmallStrainIsotropicLinearElasticModel.cxx
      test_WorkDispatcher.cxx
      test_J2IsotropicYieldSurface.cxx
      test_LinearIsotropicHardening.cxx
      test_PerzynaInelasticFlowRate.cxx
      test_AssociativeInelasticFlowDirection.cxx
      test_AssociativeInelasticHardening.cxx
      test_InelasticModel.cxx
      test_IntegratedRateFormModel.cxx
      SampleNonlinearSystems.cxx
      test_NewtonNonlinearSolver.cxx
      test_SolveImplicitFunctionModel.cxx
      UniaxialStrainStructuralDriver.cxx
      regression_perzyna.cxx
)

# Libraries and headers
target_link_libraries(unit_tests Catch2::Catch2)
target_link_libraries(unit_tests batchedmat)
target_include_directories(unit_tests PUBLIC "${CMAKE_SOURCE_DIR}/include")

# For benchmarking
option(BENCHMARK "include benchmark tests" OFF)

if(BENCHMARK)
      target_compile_definitions(unit_tests PUBLIC ENABLE_BENCHMARK)
endif()

# Register with CTest
include(CTest)
include(Catch)

catch_discover_tests(unit_tests)
