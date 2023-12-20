[Drivers]
  [unit]
    type = NewModelUnitTest
    model = 'D'
    batch_shape = '(1)'
    input_scalar_names = 'forces/T'
    input_scalar_values = '300'
    output_symr2_names = 'D'
    output_symr2_values = 'DT'
    check_second_derivatives = true
  []
[]

[Models]
  [D]
    type = SR2LinearInterpolation
    argument = 'forces/T'
    abscissa = 'T'
    ordinate = 'D'
  []
[]

[Tensors]
  [T]
    type = LinspaceScalar
    start = 273.15
    end = 2000
    nstep = 100
    dim = 0
  []
  [d0]
    type = FullScalar
    value = 1
  []
  [D0]
    type = FillSR2
    values = 'd0'
  []
  [d1]
    type = FullScalar
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
    value = 1.4509077221530537
  []
  [DT]
    type = FillSR2
    values = 'dT'
  []
[]
