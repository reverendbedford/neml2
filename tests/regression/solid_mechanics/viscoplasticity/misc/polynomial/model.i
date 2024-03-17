[Tensors]
  [end_time]
    type = LogspaceScalar
    start = -3
    end = -3
    nstep = 20
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = 100
  []
  [start_temperature]
    type = LinspaceScalar
    start = 100
    end = 1000
    nstep = 20
  []
  [end_temperature]
    type = LinspaceScalar
    start = 200
    end = 1500
    nstep = 20
  []
  [temperatures]
    type = LinspaceScalar
    start = start_temperature
    end = end_temperature
    nstep = 100
  []
  [exx]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.1
  []
  [eyy]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [ezz]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [max_strain]
    type = FillSR2
    values = 'exx eyy ezz'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = 100
  []
[]

[Drivers]
  [driver]
    type = SolidMechanicsDriver
    model = 'model'
    times = 'times'
    prescribed_strains = 'strains'
    prescribed_temperatures = 'temperatures'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-8
    rel_tol = 1e-6
  []
[]

[Models]
  #####################################################################################
  # Compute the invariant plastic flow direction since we are doing J2 radial return
  #####################################################################################
  [trial_elastic_strain]
    type = ElasticStrain
    plastic_strain = 'old_state/Ep'
    elastic_strain = 'state/Ee'
  []
  [cauchy_stress]
    type = LinearIsotropicElasticity
    youngs_modulus = 1e5
    poisson_ratio = 0.3
    strain = 'state/Ee'
    stress = 'state/S'
  []
  [flow_direction]
    type = J2FlowDirection
    mandel_stress = 'state/S'
    flow_direction = 'forces/N'
  []
  [trial_state]
    type = ComposedModel
    models = 'trial_elastic_strain cauchy_stress flow_direction'
  []

  #####################################################################################
  # Stress update
  #####################################################################################
  [ep_rate]
    type = ScalarStateRate
    state = 'ep'
  []
  [plastic_strain_rate]
    type = AssociativePlasticFlow
    flow_direction = 'forces/N'
    flow_rate = 'state/ep_rate'
    plastic_strain_rate = 'state/Ep_rate'
  []
  [plastic_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'Ep'
  []
  [elastic_strain]
    type = ElasticStrain
    plastic_strain = 'state/Ep'
    elastic_strain = 'state/Ee'
  []
  [stress_update]
    type = ComposedModel
    models = 'ep_rate plastic_strain_rate plastic_strain elastic_strain cauchy_stress'
  []

  #####################################################################################
  # Compute the rates of equivalent plastic strain and internal variables
  #####################################################################################
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/S'
    invariant = 'state/s'
  []
  [rom]
    type = TabulatedPolynomialModel
    von_mises_stress = 'state/s'
    temperature = 'forces/T'
    internal_state_1 = 'state/s1'
    internal_state_2 = 'state/s2'
    equivalent_plastic_strain_rate = 'state/ep_rate'
    internal_state_1_rate = 'state/s1_rate'
    internal_state_2_rate = 'state/s2_rate'
    use_AD_first_derivative = true
    use_AD_second_derivative = true
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'ep'
  []
  [integrate_s1]
    type = ScalarBackwardEulerTimeIntegration
    variable = 's1'
  []
  [integrate_s2]
    type = ScalarBackwardEulerTimeIntegration
    variable = 's2'
  []
  [rate]
    type = ComposedModel
    models = "stress_update vonmises rom
              integrate_ep integrate_s1 integrate_s2"
  []
  [radial_return]
    type = ImplicitUpdate
    implicit_model = 'rate'
    solver = 'newton'
  []

  #####################################################################################
  # Put the models together
  #####################################################################################
  [model]
    type = ComposedModel
    models = 'trial_state radial_return stress_update'
    additional_outputs = 'state/ep'
  []
[]
