[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    nbatch = 5
    input_scalar_names = 'forces/T'
    input_scalar_values = '300'
    output_scalar_names = 'E'
    output_scalar_values = '188911.6020499754'
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = ScalarLinearInterpolation
    parameter = 'E'
    argument = 'forces/T'
    abscissa = 'T'
    ordinate = 'E'
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
  [E0]
    type = FullScalar
    batch_shape = '(1)'
    value = 1.9e5
  []
  [E1]
    type = FullScalar
    batch_shape = '(1)'
    value = 1.2e5
  []
  [E]
    type = LinspaceScalar
    start = 'E0'
    end = 'E1'
    nstep = 100
    dim = 0
  []
[]
