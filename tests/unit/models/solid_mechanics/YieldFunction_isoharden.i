[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/h'
    input_Scalar_values = '20'
    input_SR2_names = 'state/internal/M'
    input_SR2_values = 'M'
    output_Scalar_names = 'state/internal/fp'
    output_Scalar_values = '83.5577'
    derivative_abs_tol = 1e-06
    check_second_derivatives = true
  []
[]

[Tensors]
  [M]
    type = FillSR2
    values = '40 120 80 10 10 90'
  []
[]

[Models]
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/M'
    invariant = 'state/internal/s'
  []
  [yield]
    type = YieldFunction
    yield_stress = 50
    isotropic_hardening = 'state/internal/h'
  []
  [model]
    type = ComposedModel
    models = 'vonmises yield'
  []
[]
