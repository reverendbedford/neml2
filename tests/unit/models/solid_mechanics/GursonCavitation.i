[Drivers]
  [unit]
    type = NewModelUnitTest
    model = 'model'
    batch_shape = '(10)'
    input_scalar_names = 'state/internal/f'
    input_scalar_values = '0.1'
    input_symr2_names = 'state/internal/Ep_rate'
    input_symr2_values = 'Epr'
    output_scalar_names = 'state/internal/f_rate'
    output_scalar_values = '0.1125'
    check_second_derivatives = true
  []
[]

[Tensors]
  [Epr]
    type = FillSR2
    values = '0.1 0.05 -0.025 0.15 -0.2 0.5'
  []
[]

[Models]
  [model]
    type = GursonCavitation
  []
[]
