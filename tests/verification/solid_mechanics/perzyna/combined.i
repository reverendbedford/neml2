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
    K = 2500
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
    yield_stress = 10
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
    eta = 500
    n = 5
  []
  [Eprate]
    type = PlasticStrainRate
  []
  [rate]
    type = ComposedModel
    models = 'Erate Eerate elasticity mandel_stress isoharden kinharden yield direction eeprate eprate hrate Eprate'
  []
  [implicit_rate]
    type = ImplicitTimeIntegration
    rate = rate
  []
  [model]
    type = ImplicitUpdate
    implicit_model = implicit_rate
    solver = newton
  []
[]
