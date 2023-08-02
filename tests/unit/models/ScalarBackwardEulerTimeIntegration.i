[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    nbatch = 10
    input_scalar_names = 'state/foo_rate state/foo old_state/foo forces/t old_forces/t'
    input_scalar_values = '-0.3 1.1 0 1.3 1.1'
    output_scalar_names = 'residual/foo'
    output_scalar_values = '1.16'
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'foo'
  []
[]
