[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '(10)'
    input_scalar_names = 'params/E params/nu'
    input_scalar_values = '100000.0 0.3'
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
    values = "134615.3846153846 57692.30769230767 57692.30769230767 0.0 0.0 0.0 57692.30769230767 "
             "134615.3846153846 57692.30769230767 0.0 0.0 0.0 57692.30769230767 57692.30769230767 "
             "134615.3846153846 0.0 0.0 0.0 0.0 0.0 0.0 76923.07692307692 0.0 0.0 0.0 0.0 0.0 0.0 "
             "76923.07692307692 0.0 0.0 0.0 0.0 0.0 0.0 76923.07692307692"
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
  [p]
    type = IsotropicElasticityTensor
    p1 = 'E'
    p1_type = 'youngs_modulus'
    p2 = 'nu'
    p2_type = 'poissons_ratio'
  []
  [model]
    type = ComposedModel
    models = 'p'
  []
[]
