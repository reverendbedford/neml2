[Solvers]
  [newton]
    type = NewtonNonlinearSolver
  []
[]

[Models]
  [Ee]
    type = ElasticStrain
  []
  [S]
    type = CauchyStressFromElasticStrain
    E = 120000
    nu = 0.3
  []
  [M]
    type = IsotropicMandelStress
  []
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 10.0
  []
  [j2]
    type = J2StressMeasure
  []
  [f]
    type = IsotropicHardeningYieldFunction
    stress_measure = j2
    yield_stress = 100
  []
  [Np]
    type = AssociativePlasticFlowDirection
    yield_function = f
  []
  [Eprate]
    type = PlasticStrainRate
  []
  [integrate_Ep]
    type = SymR2BackwardEulerTimeIntegration
    rate_variable = plastic_strain_rate
    variable = plastic_strain
  []
  [consistency]
    type = RateIndependentPlasticFlowConstraint
  []
  [surface]
    type = ComposedModel
    models = 'Ee S M Np Eprate integrate_Ep isoharden f consistency'
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = surface
    solver = newton
  []
  [model]
    type = ComposedModel
    models = 'return_map Ee S'
  []
[]
