[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '()'
    input_scalar_names = 'params/E params/nu params/mu'
    input_scalar_values = '100000.0 0.3 60000.0'
    output_ssr4_names = 'parameters/p'
    output_ssr4_values = 'p_correct'
    check_AD_first_derivatives = false
    check_AD_second_derivatives = false
    check_AD_derivatives = false
  []
[]

[Tensors]
  [p_correct]
    type = SSR4
    values = "134615.38461538462 57692.307692307695 57692.307692307695 0.0 0.0 0.0 57692.307692307695 134615.38461538462 57692.307692307695 0.0 0.0 0.0 57692.307692307695 57692.307692307695 134615.38461538462 0.0 0.0 0.0 0.0 0.0 0.0 120000.0 0.0 0.0 0.0 0.0 0.0 0.0 120000.0 0.0 0.0 0.0 0.0 0.0 0.0 120000.0"
  []
[]

[Models]
  [E]
    type = ScalarInputParameter
    from = 'params/E'
  []
  [nu]
    type = ScalarInputParameter
    from = 'params/nu'
  []
  [mu]
    type = ScalarInputParameter
    from = 'params/mu'
  []
  [p]
    type = CubicElasticityTensor
    p1 = 'E'
    p1_type = 'youngs_modulus'
    p2 = 'nu'
    p2_type = 'poissons_ratio'
    p3 = 'mu'
    p3_type = 'shear_modulus'
  []
  [model]
    type = ComposedModel
    models = 'p'
  []
[]
