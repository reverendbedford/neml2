[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    batch_shape = '(10)'
    output_scalar_names = 'parameters/E'
    output_scalar_values = 'T'
    check_second_derivatives = true
  []
[]

[Models]
  [E]
    type = ScalarConstantParameter
    value = 'T'
  []
[]

[Tensors]
  [T]
    type = FullScalar
    value = 20.0
  []
[]
