[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    nbatch = 10
    input_symr2_names = 'state/internal/M state/internal/X'
    input_symr2_values = 'M X'
    output_symr2_names = 'state/internal/O'
    output_symr2_values = 'O'
  []
[]

[Tensors]
  [M]
    type = InitializedSymR2
    values = '100 110 100 50 40 30'
  []
  [X]
    type = InitializedSymR2
    values = '50 -10 20 40 30 -60'
  []
  [O]
    type = InitializedSymR2
    values = '50 120 80 10 10 90'
  []
[]

[Models]
  [model]
    type = OverStress
  []
[]
