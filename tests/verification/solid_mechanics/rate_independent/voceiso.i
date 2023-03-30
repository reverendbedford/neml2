[Solvers]
  [newton]
    type = NewtonNonlinearSolver
  []
[]

[Predictors]
  [simple]
    type = LinearExtrapolationPredictor
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
  [eprate]
    type = AssociativeIsotropicPlasticHardening
    yield_function = f
  []
  [Eprate]
    type = PlasticStrainRate
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    rate_variable = 'internal_state equivalent_plastic_strain_rate'
    variable = 'internal_state equivalent_plastic_strain'
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
    models = 'Ee S M Np eprate Eprate integrate_ep integrate_Ep isoharden f consistency'
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = surface
    solver = newton
    predictor = simple
    additional_outputs = 'state plastic_strain; state internal_state equivalent_plastic_strain'
  []
  [model]
    type = ComposedModel
    models = 'return_map Ee S'
  []
[]
