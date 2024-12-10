[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model1'
      batch_shape = '(3)'
      input_scalar_names = 'state/delta'
      input_scalar_values = 'delta'
      output_scalar_names = 'state/dratio'
      output_scalar_values = 'dratio1'
      check_AD_first_derivatives = false
      check_first_derivatives = true
      
      #derivatives_abs_tol = 1e-4
      #derivatives_rel_tol = 1e-4
      #output_abs_tol = 1e-4
  []
[]

[Tensors]
    [M]
        type = Scalar
        values = "0.1 0.576 0.8"
        batch_shape = '(3)'
    []
    [delta]
        type = Scalar
        values = "0.1 0.5 0.8"
        batch_shape = '(3)'
    []
    [dratio1]
        type = Scalar
        values = "0.030727922061358 0.361906637611548 0.724077343935025"
        batch_shape = '(3)'
    []
[]

[Models]
    [model1]
        type = ProductThicknessLimit
        Initial_Porosity = 0.5
        Product_Thickness_Growth_Ratio = 'M'
        Product_Thickness = 'state/delta'
        Limit_Ratio = 'state/dratio'
    []
[]