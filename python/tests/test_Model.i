[Models]
  [foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'foo'
  []
  [bar]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'bar'
  []
  [baz]
    type = ScalarLinearCombination
    from_var = 'residual/foo residual/bar'
    to_var = 'residual/foo_bar'
  []
  [model]
    type = ComposedModel
    models = 'foo bar baz'
  []
[]
