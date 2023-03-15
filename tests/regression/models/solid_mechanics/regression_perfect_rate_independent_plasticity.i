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
  [j2]
    type = J2StressMeasure
  []
  [yield]
    type = PerfectlyPlasticYieldFunction
    stress_measure = j2
    yield_stress = 5
  []
  [direction]
    type = AssociativePlasticFlowDirection
    yield_function = yield
  []
  [ri_constraint]
    type = RateIndependentPlasticFlowConstraint
  []
  [Eprate]
    type = PlasticStrainRate
  []
  [rate]
    type = ComposedModel
    models = 'Erate Eerate elasticity mandel_stress yield direction Eprate'
  []
  [implicit_rate]
    type = ImplicitTimeIntegration
    rate = rate
  []
  [consistent_model]
    type = ComposedModel
    models = 'implicit_rate ri_constraint'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = consistent_model
    solver = newton
  []
[]
