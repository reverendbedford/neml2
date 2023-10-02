[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    nbatch = 5
    input_scalar_names = 'forces/T'
    input_scalar_values = '300'
    output_symr2_names = 'D'
    output_symr2_values = 'DT'
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = SR2LinearInterpolation
    parameter = 'D'
    argument = 'forces/T'
    abscissa = 'T'
    ordinate = 'D'
  []
[]

[Tensors]
  [T0]
    type = FullScalar
    batch_shape = '(5)'
    value = 273.15
  []
  [T1]
    type = FullScalar
    batch_shape = '(5)'
    value = 2000
  []
  [T]
    type = LinspaceScalar
    start = 'T0'
    end = 'T1'
    nstep = 100
    dim = 0
  []
  [d0]
    type = FullScalar
    batch_shape = '(1)'
    value = 1
  []
  [D0]
    type = FillSR2
    values = 'd0'
  []
  [d1]
    type = FullScalar
    batch_shape = '(1)'
    value = 30
  []
  [D1]
    type = FillSR2
    values = 'd1'
  []
  [D]
    type = LinspaceSR2
    start = 'D0'
    end = 'D1'
    nstep = 100
    dim = 0
  []
  [dT]
    type = FullScalar
    batch_shape = '(5)'
    value = 1.4509077221530537
  []
  [DT]
    type = FillSR2
    values = 'dT'
  []
[]
