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
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 1.1
  []
  [j2]
    type = J2StressMeasure
  []
  [yield]
    type = YieldFunction
    stress_measure = j2
    yield_stress = 10
    with_isotropic_hardening = true
    with_kinematic_hardening = false
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
  [rate]
    type = ComposedModel
    models = 'Erate Eerate elasticity mandel_stress isoharden yield direction eprate hrate Eprate'
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
