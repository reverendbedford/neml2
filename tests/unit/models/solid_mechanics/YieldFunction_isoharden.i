[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    nbatch = 10
    input_scalar_names = 'state/internal/k'
    input_scalar_values = '20'
    input_symr2_names = 'state/internal/M'
    input_symr2_values = 'M'
    output_scalar_names = 'state/internal/fp'
    output_scalar_values = '83.5577'
    check_second_derivatives = true
    derivatives_abs_tol = 1e-06
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
    invariant = 'state/internal/sm'
  []
  [yield]
    type = YieldFunction
    yield_stress = 50
    isotropic_hardening = 'state/internal/k'
  []
  [model]
    type = ComposedModel
    models = 'vonmises yield'
  []
[]
