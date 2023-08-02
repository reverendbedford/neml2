[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    nbatch = 10
    input_symr2_names = 'state/internal/Ee'
    input_symr2_values = 'Ee'
    output_symr2_names = 'state/S'
    output_symr2_values = 'S'
  []
[]

[Tensors]
  [Ee]
    type = InitializedSymR2
    values = '0.09 0.04 -0.02'
  []
  [S]
    type = InitializedSymR2
    values = '13.2692 9.4231 4.8077'
  []
[]

[Models]
  [model]
    type = LinearElasticity
    youngs_modulus = 100
    poisson_ratio = 0.3
  []
[]
