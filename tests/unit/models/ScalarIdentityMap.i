[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '(10)'
    input_scalar_names = 'forces/A'
    input_scalar_values = '2'
    output_scalar_names = 'state/internal/C'
    output_scalar_values = '2'
  []
[]

[Models]
  [model]
    type = ScalarIdentityMap
    from_var = 'forces/A'
    to_var = 'state/internal/C'
  []
[]
