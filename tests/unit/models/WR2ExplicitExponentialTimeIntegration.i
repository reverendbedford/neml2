[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_skewr2_names = 'state/foo_rate'
    input_skewr2_values = 'w'
    input_rot_names = 'old_state/foo'
    input_rot_values = 'old_foo'
    input_scalar_names = 'forces/t old_forces/t'
    input_scalar_values = '1.3 1.1'
    output_rot_names = 'state/foo'
    output_rot_values = 'foo'
    check_AD_derivatives = false
    check_AD_second_derivatives = false
  []
[]

[Models]
  [model]
    type = WR2ExplicitExponentialTimeIntegration
    variable = 'foo'
  []
[]

[Tensors]
  [foo]
    type = FillRot
    values = '0.07598595 -0.03827791 0.05679948'
  []
  [old_foo]
    type = FillRot
    values = '0.075 -0.04 0.06'
  []
  [w]
    type = FillWR2
    values = '0.01 0.02 -0.03'
  []
[]