[Drivers]
  [unit]
    type = NewModelUnitTest
    model = 'model'
    batch_shape = '(10)'
    input_scalar_names = 'state/internal/gamma_rate state/internal/Nk'
    input_scalar_values = '0.0015 -1'
    output_scalar_names = 'state/internal/ep_rate'
    output_scalar_values = '0.0015'
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = AssociativeIsotropicPlasticHardening
  []
[]
