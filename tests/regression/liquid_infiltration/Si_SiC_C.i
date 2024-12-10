[Tensors]
    ############### Run condition ############### 
    [endtime]
        type = Scalar
        values = 1080000
        batch_shape ='(5)'
    []
    [times]
        type = LinspaceScalar
        start = 0 
        end = endtime
        nstep = 10000
    []
    [aInDotStart]
        type = Scalar
        values = 0.3
        batch_shape ='(5)'
    []
    [aInDotEnd]
        type = Scalar
        values = 0.3
        batch_shape ='(5)'
    []
    [aLInDot]
        type = LinspaceScalar
        start = aInDotStart
        end = aInDotEnd
        nstep = 10000
    []
    ############### Simulation parameters ############### 
    [M]
        type = Scalar
        values = '0.576'
        batch_shape = '(5)'
    []
    [phi0]
        type = Scalar
        values = '0.1 0.3 0.5 0.7 0.9'
        batch_shape = '(5)'
    []
    [omega_L]
        type = Scalar
        values = 1.2e-5
        batch_shape = '(5)'
    []
    [omega_P]
        type = Scalar
        values = 1.25e-5
        batch_shape = '(5)'
    []
    [p]
        type = Scalar
        values = '1.0'
        batch_shape = '(5)'
    []
    [lc]
        type = Scalar
        values = '1.0'
        batch_shape = '(5)'
    []
[]

[Drivers]
    [regression]
      type = LiquidInfiltrationDriver
      model = 'model'
      enable_AD = false
      times = 'times'
      time = 'forces/tt'

      Prescribed_Liquid_Inlet_Rate = 'aLInDot'
      Liquid_Inlet_Rate = 'forces/aLInDot'
      
      show_input_axis = true
      show_output_axis = true
      show_parameters = false
      ic_scalar_names = 'state/alpha state/delta state/h state/alphaP'
      ic_scalar_values = '1e-4 1e-2 1e-3 -0.3'
      save_as = 'try.pt'

      verbose = true
    []
    #[regression]
    #  type = TransientRegression
    #  driver = 'driver'
    #  reference = 'gold/result.pt'
    #[]
  []

[Solvers]
    [newton]
        type = NewtonWithLineSearch
        rel_tol = 1e-8
        abs_tol = 1e-10
        max_its = 100
        verbose = true
    []
[]

[Models]
    [inlet_gap]
        type = InletGap
        Product_Thickness_Growth_Ratio = 'M'
        Initial_Porosity = 'phi0'
        Product_Thickness = 'state/delta'
        Inlet_Gap = 'state/r1'
    []
    [product_growth]
        type = ProductGrowthWithLiquid
        Liquid_Molar_Volume = 'omega_L'
        Product_Height = 'state/h'
        Inlet_Gap = 'state/r1'
        Liquid_Saturation = 'state/alpha'
        Phi_Condition = 'state/pcond'
    []
    [hdot]
        type = ScalarVariableRate
        variable = 'state/h'
        time = 'forces/tt'
        rate = 'state/hdot'
    []
    [fbcond]
        type = FischerBurmeister
        Condition_A = 'state/pcond'
        Condition_B = 'state/hdot'
        Fischer_Burmeister = 'residual/h'
    []
    ############### H RESIDUAL ############### 
    [residual_h]
        type = ComposedModel
        models = 'fbcond inlet_gap product_growth hdot'
    []
    #############################################
    [product_geo]
        type = ProductGeometricRelation
        Product_Molar_Volume = 'omega_P'
        Product_Height = 'state/h'
        Inlet_Gap = 'state/r1'
        Product_Thickness = 'state/delta'
        Product_Saturation = 'state/alphaP'
    []
    [alpha_transition]
        type = SwitchingFunction
        Smooth_Degree = 100.0
        variable = 'state/h'
        switch_out = 'state/alpha_transition'
    []
    [aR_dot]
        type = ScalarVariableRate
        variable = 'state/alphaP'
        time = 'forces/tt'
        rate = 'state/aRdot'
    []
    [alpha_dot]
        type = ScalarVariableRate
        variable = 'state/alpha'
        time = 'forces/tt'
        rate = 'state/alphadot'
    []
    [mass_balance]
        type = ChemMassBalance
        In = 'forces/aLInDot'
        Switch = 'state/alpha_transition'
        Minus_Reaction = 'state/aRdot'
        Current = 'state/alphadot'
        Total = 'residual/alpha'
    []
    ############### ALPHA RESIDUAL ############### 
    [residual_alpha]
        type = ComposedModel
        models = 'product_geo inlet_gap alpha_transition aR_dot alpha_dot mass_balance'
    []
    #############################################
    [deficient_scale]
        type = LiquidDeficientPowerScale
        Liquid_Molar_Volume = 'omega_L'
        Power = 'p'
        Product_Height = 'state/h'
        Inlet_Gap = 'state/r1'
        Liquid_Saturation = 'state/alpha'
        Scale = 'state/def_scale'
    []
    [perfect_growth]
        type = LiquidProductDiffusion1D
        Liquid_Product_Density_Ratio = 0.8
        Initial_Porosity = 'phi0'
        Product_Thickness_Growth_Ratio = 'M'
        Liquid_Product_Diffusion_Coefficient = 5e-7
        Representative_Pores_Size = 'lc'

        Inlet_Gap = 'state/r1'
        Product_Thickness = 'state/delta'
        Ideal_Thickness_Growth = 'state/delta_growth'
    []
    [delta_dcrit_ratio]
        type = ProductThicknessLimit
        Initial_Porosity = 'phi0'
        Product_Thickness_Growth_Ratio = 'M'
        Product_Thickness = 'state/delta'
        Limit_Ratio = 'state/dratio'
    []
    [delta_limit]
        type = SwitchingFunction
        Smooth_Degree = 100.0
        variable = 'state/dratio'
        switch_out = 'state/dlimit'
    []
    [ddot]
        type = ScalarVariableRate
        variable = 'state/delta'
        time = 'forces/tt'
        rate = 'state/ddot'
    []
    [product_thickness_growth]
        type = ProductWidthGrowthRate
        Thickness_Rate = 'state/ddot'
        Scale = 'state/def_scale'
        Ideal_Thickness_Growth = 'state/delta_growth'
        Switch = 'state/dlimit'
        Residual_Delta = 'residual/delta'
    []
    ############### DELTA RESIDUAL ############### 
    [residual_delta]
        type = ComposedModel
        models = 'deficient_scale inlet_gap perfect_growth delta_dcrit_ratio delta_limit ddot product_thickness_growth'
    []
    #############################################
    [model_residual]
        type = ComposedModel
        models = 'residual_h residual_alpha residual_delta'
        automatic_scaling = true
    []
    [model_update]
        type = ImplicitUpdate
        implicit_model = 'model_residual'
        solver = 'newton'
    []
    [aSiC_new]
        type = ProductGeometricRelation
        Product_Molar_Volume = 'omega_P'
        Product_Height = 'state/h'
        Inlet_Gap = 'state/r1'
        Product_Thickness = 'state/delta'
        Product_Saturation = 'state/alphaP'
    []
    [model]
        type = ComposedModel
        models = 'model_update inlet_gap aSiC_new'
        additional_outputs = 'state/delta state/h'
    []
[]