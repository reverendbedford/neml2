[Solvers]
  [newton]
    type = NewtonNonlinearSolver
  []
[]

[Models]
  [Erate]
    type = SymR2ForceRate
    force = total_strain
  []
  [Eerate]
    type = ElasticStrainRate
  []
  [elasticity]
    type = CauchyStressRateFromElasticStrainRate
    E = 1e5
    nu = 0.3
  []
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [isoharden]
    type = LinearIsotropicHardening
    K = 500
  []
  [kinharden]
    type = LinearKinematicHardening
    H = 1000
  []
  [j2]
    type = J2StressMeasure
  []
  [yield]
    type = IsotropicAndKinematicHardeningYieldFunction
    stress_measure = j2
    yield_stress = 5
  []
  [direction]
    type = AssociativePlasticFlowDirection
    yield_function = yield
  []
  [eeprate]
    type = AssociativeIsotropicPlasticHardening
    yield_function = yield
  []
  [eprate]
    type = AssociativeKinematicPlasticHardening
    yield_function = yield
  []
  [hrate]
    type = PerzynaPlasticFlowRate
    eta = 100
    n = 2
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
    rate_variable = 'internal_state plastic_strain_rate'
    variable = 'internal_state plastic_strain'
  []
  [integrate_stress]
    type = SymR2BackwardEulerTimeIntegration
    rate_variable = cauchy_stress_rate
    variable = cauchy_stress
  []
  [implicit_rate]
    type = ComposedModel
    models = 'Erate Eerate elasticity mandel_stress isoharden kinharden yield direction eeprate eprate hrate Eprate integrate_ep integrate_Ep integrate_stress'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = implicit_rate
    solver = newton
  []
[]
