[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    batch_shape = '(10)'
    output_symr2_names = 'E'
    output_symr2_values = 'T'
    check_second_derivatives = true
  []
[]

[Models]
  [E]
    type = SR2ConstantParameter
    value = 'T'
  []
[]

[Tensors]
  [T]
    type = FillSR2
    values = '-1 -4 7 -1 9 1'
  []
[]
