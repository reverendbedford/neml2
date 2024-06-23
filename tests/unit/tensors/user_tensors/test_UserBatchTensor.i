[Tensors]
  [a]
    type = BatchTensor
    values = '1 2 3 4 5 6 7 8 9 10 11 12'
    batch_shape = '(2)'
    base_shape = '(2,3)'
  []
  [b]
    type = BatchTensor
    values = '1 2 3 4 5 6 7 8 9 10 11 12'
    batch_shape = '()'
    base_shape = '(2,2,3)'
  []
  [c]
    type = BatchTensor
    values = '1 2 3 4 5 6 7 8 9 10 11 12'
    batch_shape = '(2,2,3)'
    base_shape = '()'
  []
  [d]
    type = BatchTensor
    values = '1 2 3 4 5 6'
    batch_shape = '(2)'
    base_shape = '(2,3)'
  []
  [e]
    type = BatchTensor
    values = 'unit/tensors/user_tensors/test_UserBatchTensor.csv:value'
    batch_shape = '(2)'
    base_shape = '(2,3)'
  []
[]
