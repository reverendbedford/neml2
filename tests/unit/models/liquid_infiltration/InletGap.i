[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      batch_shape = '(3)'
      input_scalar_names = 'state/delta/' #params/M params/phi0'
      input_scalar_values = 'dd'
      output_scalar_names = 'state/r1'
      output_scalar_values = 'rr'
      check_AD_first_derivatives = false
      check_first_derivatives = true
  []
[]

[Tensors]
    [dd]
        type = Scalar
        values = "0.1 0.5 0.8"
        batch_shape = '(3)'
    []
    #[mm]
    #    type = Scalar
    #    values = "0.576 0.576 0.576"
    #    batch_shape = '(3)'
    #[]
    #[p0]
    #    type = Scalar
    #    values = "0.5 0.5 0.5"
    #    batch_shape = '(3)'
    #[]
    [rr]
        type = Scalar
        values = "0.70134678 0.56310678 0.33846678"
        batch_shape = '(3)'
    []
[]

[Models]
    #[P0]
    #    type = ScalarInputParameter
    #    from = 'params/phi0'
    #[]
    #[MM]
    #    type = ScalarInputParameter
    #    from = 'params/M'
    #[]
    [model]
        type = InletGap
        Product_Thickness_Growth_Ratio = 0.576
        Initial_Porosity = 0.5
        Product_Thickness = 'state/delta'
        Inlet_Gap = 'state/r1'
    []
    #[model]
    #    type = ComposedModel
    #    models = 'model0'
    #[]
[]