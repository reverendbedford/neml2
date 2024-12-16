[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'state/S'
    input_SR2_values = 'S'
    output_SR2_names = 'state/internal/M'
    output_SR2_values = 'M'
  []
[]

[Tensors]
  [S]
    type = FillSR2
    values = '50 -10 20 40 30 -60'
  []
  [M]
    type = FillSR2
    values = '50 -10 20 40 30 -60'
  []
[]

[Models]
  [model]
    type = IsotropicMandelStress
  []
[]
