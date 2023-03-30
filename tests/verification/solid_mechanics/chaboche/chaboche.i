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
  [hrate]
    type = PerzynaPlasticFlowRate
    eta = 155.22903539478642 # 200 * (2/3)^(5/8)
    n = 4
  []
  [Eprate]
    type = PlasticStrainRate
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    rate_variable = 'internal_state equivalent_plastic_strain_rate'
    variable = 'internal_state equivalent_plastic_strain'
  []
  [integrate_X1]
    type = SymR2BackwardEulerTimeIntegration
    rate_variable = 'internal_state backstress_1_rate'
    variable = 'internal_state backstress_1'
  []
  [integrate_X2]
    type = SymR2BackwardEulerTimeIntegration
    rate_variable = 'internal_state backstress_2_rate'
    variable = 'internal_state backstress_2'
  []
  [integrate_stress]
    type = SymR2BackwardEulerTimeIntegration
    rate_variable = cauchy_stress_rate
    variable = cauchy_stress
  []
  [implicit_rate]
    type = ComposedModel
    models = 'Erate Eerate elasticity mandel_stress isoharden chaboche1 chaboche2 kinharden yield direction eeprate hrate Eprate integrate_ep integrate_X1 integrate_X2 integrate_stress'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = implicit_rate
    solver = newton
  []
[]
