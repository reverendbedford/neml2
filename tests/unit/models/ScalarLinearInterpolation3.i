[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '(2,5,2)'
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
  [T]
    type = LinspaceScalar
    start = 273.15
    end = 2000
    nstep = 100
    dim = 0
  []
  [E0]
    type = FullScalar
    batch_shape = '(5,1)'
    value = 1.9e5
  []
  [E1]
    type = FullScalar
    batch_shape = '(5,1)'
    value = 1.2e5
  []
  [E]
    type = LinspaceScalar
    start = 'E0'
    end = 'E1'
    nstep = 100
    dim = 2
  []
[]
