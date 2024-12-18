[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/A state/substate/B params/c_A params/c_B'
    input_Scalar_values = '3 2 1 2'
    output_Scalar_names = 'state/outsub/C'
    output_Scalar_values = '7'
  []
[]

[Models]
  [c_A]
    type = ScalarInputParameter
    from = 'params/c_A'
  []
  [c_B]
    type = ScalarInputParameter
    from = 'params/c_B'
  []
  [model0]
    type = ScalarLinearCombination
    from_var = 'state/A state/substate/B'
    to_var = 'state/outsub/C'
    coefficients = 'c_A c_B'
    coefficient_as_parameter = 'true true'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
