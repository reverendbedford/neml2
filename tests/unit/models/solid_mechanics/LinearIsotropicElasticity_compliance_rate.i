[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'state/S_rate'
    input_SR2_values = 'S_rate'
    output_SR2_names = 'state/internal/Ee_rate'
    output_SR2_values = 'Ee_rate'
  []
[]

[Tensors]
  [Ee_rate]
    type = FillSR2
    values = '0.09 0.04 -0.02'
  []
  [S_rate]
    type = FillSR2
    values = '13.2692 9.4231 4.8077'
  []
[]

[Models]
  [model]
    type = LinearIsotropicElasticity
    youngs_modulus = 100
    poisson_ratio = 0.3
    rate_form = true
    compliance = true
  []
[]
