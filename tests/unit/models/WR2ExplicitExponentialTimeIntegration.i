[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_WR2_names = 'state/foo_rate'
    input_WR2_values = 'w'
    input_Rot_names = 'old_state/foo'
    input_Rot_values = 'old_foo'
    input_Scalar_names = 'forces/t old_forces/t'
    input_Scalar_values = '1.3 1.1'
    output_Rot_names = 'state/foo'
    output_Rot_values = 'foo'
  []
[]

[Models]
  [model]
    type = WR2ExplicitExponentialTimeIntegration
    variable = 'state/foo'
  []
[]

[Tensors]
  [foo]
    type = FillRot
    values = '0.03789409 -0.01908914 0.02832582'
  []
  [old_foo]
    type = FillRot
    values = '0.03739906 -0.01994617 0.02991925'
  []
  [w]
    type = FillWR2
    values = '0.01 0.02 -0.03'
  []
[]
