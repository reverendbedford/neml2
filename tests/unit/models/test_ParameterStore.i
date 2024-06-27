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
    youngs_modulus = 'E1'
    poisson_ratio = 0.3
    stress = 'state/S1'
  []
  [elasticity2]
    type = LinearIsotropicElasticity
    youngs_modulus = 'E2'
    poisson_ratio = 0.3
    stress = 'state/S2'
  []
  [elasticity2_another]
    type = LinearIsotropicElasticity
    youngs_modulus = 'E2'
    poisson_ratio = 0.3
    stress = 'state/S2_another'
  []
  [elasticity3]
    type = LinearIsotropicElasticity
    youngs_modulus = 'E3'
    poisson_ratio = 0.3
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
