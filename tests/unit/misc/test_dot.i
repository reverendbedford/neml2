[Solvers]
  [newton]
    type = NewtonNonlinearSolver
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
  [model]
    type = ComposedModel
    models = 'rate integrate_foo integrate_bar integrate_baz'
  []
[]
