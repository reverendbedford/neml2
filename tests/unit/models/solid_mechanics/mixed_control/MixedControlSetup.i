[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '()'
    input_symr2_names = 'state/mixed_state old_state/mixed_state'
    input_symr2_values = 'mvals old_mvals'
    output_symr2_names = 'state/S forces/E'
    output_symr2_values = 'stress strain'
  []
[]

[Tensors]
  [vals]
    type = FillSR2
    values = '-50.0 0.1 0.15 -25.0 30.0 -0.05'
  []
  [mvals]
    type = FillSR2
    values = '0.1 100.0 20.0 -0.05 -0.025 50.0'
  []
  [old_mvals]
    type = FillSR2
    values = '0.9 90.0 2.0 -0.5 -0.25 5.0'
  []
  [stress]
    type = FillSR2
    values = '-50.0 100.0 20.0 -25.0 30.0 50.0'
  []
  [strain]
    type = FillSR2
    values = '0.1 0.1 0.15 -0.05 -0.025 -0.05'
  []
[]

[Models]
  [model]
    type = MixedControlSetup
    control = 'stress strain strain stress stress strain'
    fixed_values = vals
  []
[]
