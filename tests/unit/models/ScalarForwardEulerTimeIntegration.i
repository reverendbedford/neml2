[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/foo_rate old_state/foo forces/t old_forces/t'
    input_Scalar_values = '-0.3 0 1.3 1.1'
    output_Scalar_names = 'state/foo'
    output_Scalar_values = '-0.06'
  []
[]

[Models]
  [model]
    type = ScalarForwardEulerTimeIntegration
    variable = 'state/foo'
  []
[]
