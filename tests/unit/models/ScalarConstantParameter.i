[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    output_Scalar_names = 'parameters/E'
    output_Scalar_values = 'T'
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
