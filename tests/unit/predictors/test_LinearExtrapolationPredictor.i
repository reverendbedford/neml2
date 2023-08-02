[Solvers]
  [newton]
    type = NewtonNonlinearSolver
    abs_tol = 1e-10
    rel_tol = 1e-08
    max_its = 100
  []
[]

[Predictors]
  [simple]
    type = LinearExtrapolationPredictor
  []
[]

[Models]
  [rate]
    type = SampleRateModel
  []
  [integrate_foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'foo'
  []
  [integrate_bar]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'bar'
  []
  [integrate_baz]
    type = SymR2BackwardEulerTimeIntegration
    variable = 'baz'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'rate integrate_foo integrate_bar integrate_baz'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
    predictor = 'simple'
  []
[]
