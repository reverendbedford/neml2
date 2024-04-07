[Models]
  [integrate_foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'foo'
  []
  [integrate_bar]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'bar'
  []
  [residual_sum]
    type = ScalarSumModel
    from_var = 'residual/foo residual/bar'
    to_var = 'residual/foo_bar'
  []
  [model]
    type = ComposedModel
    models = 'integrate_foo integrate_bar residual_sum'
  []
[]
