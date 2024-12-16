[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/ep'
    input_Scalar_values = '0.1'
    output_Scalar_names = 'state/internal/k'
    output_Scalar_values = '100'
  []
[]

[Models]
  [model]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
[]
