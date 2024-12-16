[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/gamma_rate state/internal/Nk'
    input_Scalar_values = '0.0015 -1'
    output_Scalar_names = 'state/internal/ep_rate'
    output_Scalar_values = '0.0015'
  []
[]

[Models]
  [model]
    type = AssociativeIsotropicPlasticHardening
  []
[]
