[Tensors]
  [X_values]
    type = Scalar
    values = '0.0 1.0 2.0 3.0'
    batch_shape = '(4)'
  []
  [Y_values]
    type = Scalar
    values = '2.0 -1.0 5.0 10.0'
    batch_shape = '(4)'
  []
[]

[Models]
  [model]
    type = ScalarLinearInterpolation
    argument = 'forces/inp'
    abscissa = 'X_values'
    ordinate = 'Y_values'
  []
[]
