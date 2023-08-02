[Tensors]
  [foo]
    type = InitializedSymR2
    values = '1 2 3'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    nbatch = 10
    input_symr2_names = 'forces/A'
    input_symr2_values = 'foo'
    output_symr2_names = 'state/internal/C'
    output_symr2_values = 'foo'
  []
[]

[Models]
  [model]
    type = SymR2IdentityMap
    from_var = 'forces/A'
    to_var = 'state/internal/C'
  []
[]
