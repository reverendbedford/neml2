[Settings]
  parameter_name_separator = '::'
[]

[Models]
  [E1]
    type = ScalarConstantParameter
    value = 1e3
  []
  [E2]
    type = ScalarConstantParameter
    value = 2e3
  []
  [E3]
    type = ScalarConstantParameter
    value = 3e3
  []
  [elasticity1]
    type = LinearIsotropicElasticity
    coefficients = 'E1 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    stress = 'state/S1'
  []
  [elasticity2]
    type = LinearIsotropicElasticity
    coefficients = 'E2 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    stress = 'state/S2'
  []
  [elasticity2_another]
    type = LinearIsotropicElasticity
    coefficients = 'E2 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    stress = 'state/S2_another'
  []
  [elasticity3]
    type = LinearIsotropicElasticity
    coefficients = 'E3 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    stress = 'state/S3'
  []
  [model1]
    type = ComposedModel
    models = 'elasticity1 elasticity2'
  []
  [model2]
    type = ComposedModel
    models = 'elasticity2_another elasticity3'
  []
  [model]
    type = ComposedModel
    models = 'model1 model2'
  []
[]
