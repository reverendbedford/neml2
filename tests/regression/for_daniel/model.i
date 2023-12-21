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
    type = ThermalStructuralDriver
    model = 'model'
    times = 'times'
    prescribed_temperatures = 'temperatures'
    prescribed_strains = 'strains'
    verbose = true
  []
[]

[Solvers]
  [newton]
    type = NewtonNonlinearSolver
    verbose = true
    abs_tol = 1e-8
    rel_tol = 1e-6
  []
[]

[Models]
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [flow_rate]
    type = DanielFlowRate
  []
  [flow_direction]
    type = J2FlowDirection
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [Erate]
    type = SR2ForceRate
    force = 'E'
  []
  [Eerate]
    type = ElasticStrain
    rate_form = true
  []
  [elasticity]
    type = LinearIsotropicElasticity
    youngs_modulus = 1e5
    poisson_ratio = 0.3
    rate_form = true
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'S'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'mandel_stress flow_rate flow_direction Eprate Erate Eerate elasticity integrate_stress'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
[]
