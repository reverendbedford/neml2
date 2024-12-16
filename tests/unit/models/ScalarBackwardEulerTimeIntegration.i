[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/foo_rate state/foo old_state/foo forces/t old_forces/t'
    input_Scalar_values = '-0.3 1.1 0 1.3 1.1'
    output_Scalar_names = 'residual/foo'
    output_Scalar_values = '1.16'
  []
[]

[Models]
  [model]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/foo'
  []
[]
