[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    nbatch = 10
    input_symr2_names = 'state/internal/Ee state/internal/Ep'
    input_symr2_values = 'Ee Ep'
    output_symr2_names = 'state/E'
    output_symr2_values = 'E'
  []
[]

[Tensors]
  [E]
    type = FillSR2
    values = '0.100 0.110 0.100 0.050 0.040 0.030'
  []
  [Ep]
    type = FillSR2
    values = '0.050 -0.010 0.020 0.040 0.030 -0.060'
  []
  [Ee]
    type = FillSR2
    values = '0.050 0.120 0.080 0.010 0.010 0.090'
  []
[]

[Models]
  [model]
    type = TotalStrain
  []
[]
