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
    K = 1000
  []
  [j2]
    type = J2StressMeasure
  []
  [yield]
    type = IsotropicHardeningYieldFunction
    stress_measure = j2
    yield_stress = 5
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
    eta = 100
    n = 2
  []
  [Eprate]
    type = PlasticStrainRate
  []
  [integrate_ep]
    type = ScalarImplicitTimeIntegration
    rate_variable = 'internal_state equivalent_plastic_strain_rate'
    variable = 'internal_state equivalent_plastic_strain'
  []
  [integrate_stress]
    type = SymR2ImplicitTimeIntegration
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
