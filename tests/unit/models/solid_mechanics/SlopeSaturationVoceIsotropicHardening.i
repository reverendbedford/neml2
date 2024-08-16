[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '(10)'
    input_scalar_names = 'state/internal/ep'
    input_scalar_values = '0.1'
    output_scalar_names = 'state/internal/k'
    output_scalar_values = '10.416586470347177'
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = SlopeSaturationVoceIsotropicHardening
    saturated_hardening = 100
    initial_hardening_rate = 110.0
  []
[]
