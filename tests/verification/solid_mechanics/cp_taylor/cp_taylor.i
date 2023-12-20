[Tensors]
  [times]
    type = VTestTimeSeries
    vtest = 'cp_taylor.vtest'
    variable = 'time'
    variable_type = 'SCALAR'
    expand_batch = 5
  []
  [deformation_rate]
    type = VTestTimeSeries
    vtest = 'cp_taylor.vtest'
    variable = 'deformation_rate'
    variable_type = 'SYMR2'
    expand_batch = 5
  []
  [stresses]
    type = VTestTimeSeries
    vtest = 'cp_taylor.vtest'
    variable = 'stress'
    variable_type = 'SYMR2'
  []
  [vorticity]
    type = VTestTimeSeries
    vtest = 'cp_taylor.vtest'
    variable = 'vorticity'
    variable_type = 'WR2'
    expand_batch = 5
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
  [initial_orientation]
    type = Orientation
    input_type = "euler_angles"
    angle_convention = "kocks"
    angle_type = "degrees"
    values = '45 50 51
              75 50 10
              10 5  60
              17 18 19
              30 60 90'
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
  []
  [verification]
    type = VTestVerification
    driver = 'driver'
    variables = 'output.state/internal/cauchy_stress'
    references = 'stresses'
    taylor_average = true
    # Looser tolerances here are because the NEML(1) model was generated with lagged, explict
    # integration on the orientations
    atol = 1.0
    rtol = 1e-3
  []
[]

[Solvers]
  [newton]
    type = NewtonNonlinearSolver
    linesearch = true
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
  [elasticity]
    type = LinearIsotropicElasticity
    youngs_modulus = 1e5
    poisson_ratio = 0.25
    strain = "state/elastic_strain"
    stress = "state/internal/cauchy_stress"
  []
  [resolved_shear]
    type = ResolvedShear
  []
  [elastic_stretch]
    type = ElasticStrainRate
  []
  [plastic_spin]
    type = PlasticVorticity
  []
  [plastic_deformation_rate]
    type = PlasticDeformationRate
  []
  [orientation_rate]
    type = OrientationRate
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

  [implicit_rate]
    type = ComposedModel
    models = 'elasticity orientation_rate resolved_shear elastic_stretch plastic_deformation_rate plastic_spin sum_slip_rates slip_rule slip_strength voce_hardening integrate_slip_hardening integrate_elastic_strain integrate_orientation'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
    additional_outputs = 'state/elastic_strain'
  []
  [model_with_stress]
    type = ComposedModel
    models = 'model elasticity'
  []
[]