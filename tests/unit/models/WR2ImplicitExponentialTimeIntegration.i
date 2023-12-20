[Drivers]
  [unit]
    type = NewModelUnitTest
    model = 'model'
    input_skewr2_names = 'state/foo_rate'
    input_skewr2_values = 'w'
    input_rot_names = 'state/foo old_state/foo'
    input_rot_values = 'foo old_foo'
    input_scalar_names = 'forces/t old_forces/t'
    input_scalar_values = '1.3 1.1'
    output_rot_names = 'residual/foo'
    output_rot_values = 'res'
    check_AD_derivatives = false
    check_AD_second_derivatives = false
  []
[]

[Models]
  [model]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'foo'
  []
[]

[Tensors]
  [res]
    type = FillRot
    values = '-0.06598595 -0.01172209 0.01320052'
  []
  [foo]
    type = FillRot
    values = '0.01 -0.05 0.07'
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
