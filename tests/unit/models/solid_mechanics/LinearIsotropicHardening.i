[Drivers]
  [unit]
    type = NewModelUnitTest
    model = 'model'
    batch_shape = '(10)'
    input_scalar_names = 'state/internal/ep'
    input_scalar_values = '0.1'
    output_scalar_names = 'state/internal/k'
    output_scalar_values = '100'
  []
[]

[Models]
  [model]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
[]
