[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '()'
    input_symr2_names = 'state/mixed_state old_state/mixed_state forces/fixed_values old_forces/fixed_values forces/control old_forces/control'
    input_symr2_values = 'mvals old_mvals vals old_vals control old_control'
    output_symr2_names = 'state/S forces/E old_state/S old_forces/E'
    output_symr2_values = 'stress strain old_stress old_strain'
  []
[]

[Tensors]
  [control]
    type = FillSR2
    values = '1.0 0.0 0.0 1.0 1.0 0.0'
  []

  [old_control]
    type = FillSR2
    values = '1.0 0.0 0.0 1.0 1.0 0.0'
  []

  [vals]
    type = FillSR2
    values = '-50.0 0.1 0.15 -25.0 30.0 -0.05'
  []
  [old_vals]
    type = FillSR2
    values = '-5.0 0.15 0.5 -2.0 31.0 -0.25'
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

  [old_stress]
    type = FillSR2
    values = '-5.0 90.0 2.0 -2.0 31.0 5.0'
  []
  [old_strain]
    type = FillSR2
    values = '0.9 0.15 0.5 -0.5 -0.25 -0.25'
  []
[]

[Models]
  [model]
    type = MixedControlSetup
  []
[]
