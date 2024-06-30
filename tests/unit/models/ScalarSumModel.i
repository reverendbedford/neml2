[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '(10)'
    input_scalar_names = 'state/A state/substate/B'
    input_scalar_values = '3 2'
    output_scalar_names = 'state/outsub/C'
    output_scalar_values = '5'
    check_inplace = false
    check_concatenation = true
    check_AD_first_derivatives = false
    check_AD_second_derivatives = false
    check_AD_derivatives = false
    check_parameter_derivatives = false
  []
[]

[Models]
  [model]
    type = ScalarSumModel
    from_var = 'state/A state/substate/B'
    to_var = 'state/outsub/C'
  []
[]
