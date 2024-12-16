[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'forces/foo old_forces/foo forces/t old_forces/t'
    input_Scalar_values = '-0.3 0 1.3 1.1'
    output_Scalar_names = 'forces/foo_rate'
    output_Scalar_values = '-1.5'
  []
[]

[Models]
  [model]
    type = ScalarVariableRate
    variable = 'forces/foo'
    rate = 'forces/foo_rate'
  []
[]
