# Add all the source files
add_library(batchedmat
      utils.cxx
      Scalar.cxx
      SymR2.cxx
      SymSymR4.cxx
      StateFunction.cxx
      StateBase.cxx
      State.cxx
      StateDerivative.cxx
      StateInfo.cxx
      ConstitutiveModel.cxx
      SmallStrainMechanicalModel.cxx
      SmallStrainIsotropicLinearElasticModel.cxx
      YieldSurface.cxx
      J2IsotropicYieldSurface.cxx
      HardeningMap.cxx
      IsotropicHardeningMap.cxx
      LinearIsotropicHardening.cxx
      InelasticFlowRate.cxx
      PerzynaInelasticFlowRate.cxx
      InelasticFlowDirection.cxx
      AssociativeInelasticFlowDirection.cxx
      AssociativeInelasticHardening.cxx
      RateForm.cxx
      InelasticModel.cxx
      ImplicitFunctionModel.cxx
      IntegratedRateFormModel.cxx
      NonlinearSolver.cxx
      NewtonNonlinearSolver.cxx
      SolveImplicitFunctionModel.cxx
)

# My headers
target_include_directories(batchedmat PRIVATE "../include")

# torch headers
# torch libraries
target_link_libraries(batchedmat ${TORCH_LIBRARIES})
