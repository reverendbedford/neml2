[Drivers]
  [unit]
    type = NewModelUnitTest
    model = 'model'
    batch_shape = '(10)'
    input_scalar_names = 'forces/foo old_forces/foo forces/t old_forces/t'
    input_scalar_values = '-0.3 0 1.3 1.1'
    output_scalar_names = 'forces/foo_rate'
    output_scalar_values = '-1.5'
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = ScalarForceRate
    force = 'foo'
  []
[]
