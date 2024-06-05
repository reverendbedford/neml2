[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '(5)'
    input_scalar_names = 'forces/T'
    input_scalar_values = '1000'
    output_scalar_names = 'p'
    output_scalar_values = 'p_correct'
    check_second_derivatives = true
  []
[]

[Models]
  [p]
    type = ArrheniusParameter
    temperature = 'forces/T'
    reference_value = 'qq'
    activation_energy = 'Q'
    ideal_gas_constant = 8.314
  []
  [qq]
    type = ArrheniusParameter
    temperature = 'forces/T'
    reference_value = 'q0'
    activation_energy = 'Q'
    ideal_gas_constant = 8.314
  []
  [model]
    type = ComposedModel
    models = 'qq p'
  []
[]

[Tensors]
  [p0]
    type = LinspaceScalar
    start = 1
    end = 10
    nstep = 5
  []
  [q0]
    type = LinspaceScalar
    start = 2
    end = 5
    nstep = 5
  []
  [Q]
    type = LinspaceScalar
    start = 1e3
    end = 2e4
    nstep = 5
  []
  [p_correct]
    type = Scalar
    values = "1.57237794 0.68962443 0.27996322 0.10843634 0.04069199"
    batch_shape = '(5)'
  []
[]
