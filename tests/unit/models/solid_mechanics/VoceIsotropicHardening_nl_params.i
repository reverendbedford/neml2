[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '(10)'
    input_scalar_names = 'state/internal/ep params/R params/d'
    input_scalar_values = '0.1 100.0 1.1'
    output_scalar_names = 'state/internal/k'
    output_scalar_values = '10.416586470347177'
    check_AD_first_derivatives = false
  []
[]

[Models]
  [R]
    type = ScalarInputParameter
    from = 'params/R'
  []
  [d]
    type = ScalarInputParameter
    from = 'params/d'
  []
  [model0]
    type = VoceIsotropicHardening
    saturated_hardening = 'R'
    saturation_rate = 'd'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
