[Tensors]
  [foo]
    type = InitializedSymR2
    values = '1 2 3 4 5 6'
    # This tensor reads
    # A = [ 1 6 5
    #       6 2 4
    #       5 4 3 ]
    # I2(A) = 0.5*(36-168) = -66
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    nbatch = 1
    input_symr2_names = 'state/internal/O'
    input_symr2_values = 'foo'
    output_scalar_names = 'state/internal/I2'
    output_scalar_values = '-66'
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = SymR2Invariant
    invariant_type = 'I2'
    tensor = 'state/internal/O'
    invariant = 'state/internal/I2'
  []
[]
