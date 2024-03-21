[Tensors]
  [end_time]
    type = LinspaceScalar
    start = 1
    end = 10
    nstep = 20
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = 100
  []
  [dxx]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.1
  []
  [dyy]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [dzz]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [deformation_rate_single]
    type = FillSR2
    values = 'dxx dyy dzz'
  []
  [deformation_rate]
    type = LinspaceSR2
    start = deformation_rate_single
    end = deformation_rate_single
    nstep = 100
  []

  [w1]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.1
  []
  [w2]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [w3]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [vorticity_single]
    type = FillWR2
    values = 'w1 w2 w3'
  []
  [vorticity]
    type = LinspaceWR2
    start = vorticity_single
    end = vorticity_single
    nstep = 100
  []

  [a]
    type = Scalar
    values = '1.0'
  []
  [sdirs]
    type = FillMillerIndex
    values = '1 1 0'
  []
  [splanes]
    type = FillMillerIndex
    values = '1 1 1'
  []

  [R1]
    type = LinspaceScalar
    start = 0
    end = 0.75
    nstep = 20
  []
  [R2]
    type = LinspaceScalar
    start = 0
    end = -0.25
    nstep = 20
  []
  [R3]
    type = LinspaceScalar
    start = -0.1
    end = 0.1
    nstep = 20
  []

  [initial_orientation]
    type = FillRot
    values = 'R1 R2 R3'
    method = 'standard'
  []
[]

[Drivers]
  [driver]
    type = LargeDeformationIncrementalSolidMechanicsDriver
    model = 'model_with_stress'
    times = 'times'
    prescribed_deformation_rate = 'deformation_rate'
    prescribed_vorticity = 'vorticity'
    provide_vorticity = true
    ic_rot_names = 'state/orientation'
    ic_rot_values = 'initial_orientation'
    predictor = 'CP_PREVIOUS_STATE'
    save_as = 'result.pt'
    cp_elastic_scale = 0.1
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    max_linesearch_iterations = 5
  []
[]

[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = "a"
    slip_directions = "sdirs"
    slip_planes = "splanes"
  []
[]

[Models]
  [euler_rodrigues_old]
    type = RotationMatrix
    from = 'old_state/orientation'
    to = 'old_state/orientation_matrix'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    youngs_modulus = 1e5
    poisson_ratio = 0.25
    strain = "state/elastic_strain"
    stress = "state/internal/cauchy_stress"
  []
  [resolved_shear]
    type = ResolvedShear
    orientation = 'old_state/orientation_matrix'
  []
  [elastic_stretch]
    type = ElasticStrainRate
  []
  [plastic_spin]
    type = PlasticVorticity
    orientation = 'old_state/orientation_matrix'
  []
  [plastic_deformation_rate]
    type = PlasticDeformationRate
    orientation = "old_state/orientation_matrix"
  []
  [sum_slip_rates]
    type = SumSlipRates
  []
  [slip_rule]
    type = PowerLawSlipRule
    n = 8.0
    gamma0 = 2.0e-1
  []
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
  []
  [voce_hardening]
    type = VoceSingleSlipHardeningRule
    initial_slope = 500.0
    saturated_hardening = 50.0
  []
  [integrate_slip_hardening]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'internal/slip_hardening'
  []
  [integrate_elastic_strain]
    type = SR2BackwardEulerTimeIntegration
    variable = 'elastic_strain'
  []
  [integrate_orientation]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'orientation'
  []

  [implicit_rate_except_rotation]
    type = ComposedModel
    models = "euler_rodrigues_old elasticity resolved_shear
              elastic_stretch plastic_deformation_rate plastic_spin
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening integrate_elastic_strain"
  []
  [integrate_except_rotation]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate_except_rotation'
    solver = 'newton'
  []


  [euler_rodrigues]
    type = RotationMatrix
    from = 'state/orientation'
    to = 'state/orientation_matrix'
  []
  [elasticity_lagged]
    type = LinearIsotropicElasticity
    youngs_modulus = 1e5
    poisson_ratio = 0.25
    strain = "old_state/elastic_strain"
    stress = "old_state/internal/cauchy_stress"
  []
  [resolved_shear_lagged]
    type = ResolvedShear
    resolved_shears = "old_state/internal/resolved_shears"
    stress = "old_state/internal/cauchy_stress"
  []
  [plastic_spin_lagged]
    type = PlasticVorticity
    plastic_vorticity = "old_state/internal/plastic_vorticity"
    slip_rates = "old_state/internal/slip_rates"
  []
  [plastic_deformation_rate_lagged]
    type = PlasticDeformationRate
    plastic_deformation_rate = "old_state/internal/plastic_deformation_rate"
    slip_rates = "old_state/internal/slip_rates"
  []
  [orientation_rate]
    type = OrientationRate
    elastic_strain = "old_state/elastic_strain"
    plastic_deformation_rate = "old_state/internal/plastic_deformation_rate"
    plastic_vorticity = "old_state/internal/plastic_vorticity"
  []
  [slip_rule_lagged]
    type = PowerLawSlipRule
    n = 8.0
    gamma0 = 2.0e-1
    slip_rates = "old_state/internal/slip_rates"
    resolved_shears = "old_state/internal/resolved_shears"
    slip_strengths = "old_state/internal/slip_strengths"
  []
  [slip_strength_lagged]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
    slip_strengths = "old_state/internal/slip_strengths"
    slip_hardening = "old_state/internal/slip_hardening"
  []
  [integrate_orientation]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'orientation'
  []

  [implicit_rate_rotation]
    type = ComposedModel
    models = "euler_rodrigues elasticity_lagged resolved_shear_lagged
              plastic_deformation_rate_lagged plastic_spin_lagged
              orientation_rate
              slip_rule_lagged slip_strength_lagged
              integrate_orientation"
  []
  [integrate_rotation]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate_rotation'
    solver = 'newton'
  []


  [model]
    type = SequentialModel
    models = "integrate_except_rotation integrate_rotation"
  []

  [model_with_stress]
    type = ComposedModel
    models = 'model elasticity'
    additional_outputs = 'state/elastic_strain'
  []
[]
