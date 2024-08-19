[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '(10)'
    input_scalar_names = 'params/tau params/n'
    input_scalar_values = '120.0 2.0'
    input_symr2_names = 'state/internal/X'
    input_symr2_values = 'X'
    output_symr2_names = 'state/internal/X_recovery_rate'
    output_symr2_values = 'X_rate'
    output_abs_tol = 1.0e-4
    check_AD_first_derivatives = false
  []
[]

[Tensors]
  [X]
    type = FillSR2
    values = '-10 15 5 -7 15 20'
  []
  [X_rate]
    type = FillSR2
    values = '0.02861583 -0.04292375 -0.01430792 0.02003108 -0.04292375 -0.05723166'
  []
[]

[Models]
  [tau]
    type = ScalarInputParameter
    from = 'params/tau'
  []
  [n]
    type = ScalarInputParameter
    from = 'params/n'
  []
  [model0]
    type = KinematicHardeningStaticRecovery
    tau = 'tau'
    n = 'n'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
