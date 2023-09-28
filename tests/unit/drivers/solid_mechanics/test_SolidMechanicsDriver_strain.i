[Tensors]
  [end_time]
    type = LogspaceScalar
    start = -1
    end = 5
    nstep = 20
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
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
    save_as = 'unit/drivers/solid_mechanics/test_SolidMechanicsDriver_strain.pt'
  []
[]

[Models]
  [force_rate]
    type = SR2ForceRate
    force = 'E'
  []
  [stress_rate]
    type = LinearIsotropicElasticity
    youngs_modulus = 1e5
    poisson_ratio = 0.3
    rate_form = true
    strain = 'forces/E'
  []
  [integrate]
    type = SR2ForwardEulerTimeIntegration
    variable = 'S'
  []
  [model]
    type = ComposedModel
    models = 'force_rate stress_rate integrate'
  []
[]
