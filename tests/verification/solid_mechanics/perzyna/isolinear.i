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
    E = 124000
    nu = 0.32
  []
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [isoharden]
    type = LinearIsotropicHardening
    K = 5500
  []
  [j2]
    type = J2StressMeasure
  []
  [yield]
    type = IsotropicHardeningYieldFunction
    stress_measure = j2
    yield_stress = 10
  []
  [direction]
    type = AssociativePlasticFlowDirection
    yield_function = yield
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
    yield_function = yield
  []
  [hrate]
    type = PerzynaPlasticFlowRate
    eta = 500
    n = 5
  []
  [Eprate]
    type = PlasticStrainRate
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    rate_variable = 'internal_state equivalent_plastic_strain_rate'
    variable = 'internal_state equivalent_plastic_strain'
  []
  [integrate_stress]
    type = SymR2BackwardEulerTimeIntegration
    rate_variable = cauchy_stress_rate
    variable = cauchy_stress
  []
  [implicit_rate]
    type = ComposedModel
    models = 'Erate Eerate elasticity mandel_stress isoharden yield direction eprate hrate Eprate integrate_ep integrate_stress'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = implicit_rate
    solver = newton
  []
[]
