[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    nbatch = 1
    input_symr2_names = 'state/internal/M'
    input_symr2_values = 'M'
    input_scalar_names = 'forces/T'
    input_scalar_values = '550'
    output_scalar_names = 'state/internal/fp'
    output_scalar_values = '102.5057'
    check_second_derivatives = true
    derivatives_abs_tol = 1e-06
  []
[]

[Tensors]
  [M]
    type = FillSR2
    values = '40 120 80 10 10 90'
  []
  [T0]
    type = FullScalar
    batch_shape = '(1)'
    value = 273.15
  []
  [T1]
    type = FullScalar
    batch_shape = '(1)'
    value = 2000
  []
  [T]
    type = LinspaceScalar
    start = 'T0'
    end = 'T1'
    nstep = 10
    dim = 0
  []
  [s0_T0]
    type = FullScalar
    batch_shape = '(1)'
    value = 50
  []
  [s0_T1]
    type = FullScalar
    batch_shape = '(1)'
    value = 30
  []
  [s0_data]
    type = LinspaceScalar
    start = 's0_T0'
    end = 's0_T1'
    nstep = 10
    dim = 0
  []
[]

[Models]
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/M'
    invariant = 'state/internal/sm'
  []
  [s0]
    type = ScalarLinearInterpolation
    parameter = 's0'
    argument = 'forces/T'
    abscissa = 'T'
    ordinate = 's0_data'
  []
  [yield]
    type = YieldFunction
    yield_stress = 's0'
  []
  [model]
    type = ComposedModel
    models = 's0 vonmises yield'
  []
[]
