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
  [Eprate]
    type = PlasticStrainRate
  []
  [integrate_stress]
    type = SymR2ImplicitTimeIntegration
    rate_variable = cauchy_stress_rate
    variable = cauchy_stress
  []
  [consistency]
    type = RateIndependentPlasticFlowConstraint
  []
  [implicit_rate]
    type = ComposedModel
    models = 'Erate Eerate elasticity mandel_stress direction Eprate integrate_stress yield consistency'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = implicit_rate
    solver = newton
  []
[]
