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
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 1.2
  []
  [chaboche1]
    type = ChabochePlasticHardening
    C = 10000
    g = 100
    A = 1e-8
    a = 1.2
    backstress_suffix = '_1'
  []
  [chaboche2]
    type = ChabochePlasticHardening
    C = 1000
    g = 9
    A = 1e-10
    a = 3.2
    backstress_suffix = '_2'
  []
  [kinharden]
    type = SymR2SumModel
    from_var = 'state internal_state backstress_1; state internal_state backstress_2'
    to_var = 'state hardening_interface kinematic_hardening'
  []
  [j2]
    type = J2StressMeasure
  []
  [yield]
    type = YieldFunction
    stress_measure = j2
    yield_stress = 5
    with_isotropic_hardening = true
    with_kinematic_hardening = true
  []
  [direction]
    type = AssociativePlasticFlowDirection
    yield_function = yield
  []
  [eeprate]
    type = AssociativeIsotropicPlasticHardening
    yield_function = yield
  []
  [hrate]
    type = PerzynaPlasticFlowRate
    eta = 100
    n = 4
  []
  [Eprate]
    type = PlasticStrainRate
  []
  [rate]
    type = ComposedModel
    models = 'Erate Eerate elasticity mandel_stress isoharden chaboche1 chaboche2 kinharden yield direction eeprate hrate Eprate'
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
