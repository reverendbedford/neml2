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
    saturated_hardening = 50
    saturation_rate = 1.2
  []
  [chaboche1]
    type = ChabochePlasticHardening
    C = 5000
    g = 8.246615467370033 # 10.1 * sqrt(2/3)
    A = 1.224744871391589e-06 # 1.0e-6 * sqrt(3/2)
    a = 1.2
    backstress_suffix = '_1'
  []
  [chaboche2]
    type = ChabochePlasticHardening
    C = 1000
    g = 4.245782220824175 # 5.2 * sqrt(2/3)
    A = 1.224744871391589e-10 # 1.0e-10 * sqrt(3/2)
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
    yield_stress = 10
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
    eta = 155.22903539478642 # 200 * (2/3)^(5/8)
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
