[Tensors]
  [end_time]
    type = LogSpaceTensor
    start = -1
    end = 5
    steps = 20
  []
  [times]
    type = LinSpaceTensor
    end = end_time
    steps = 100
  []
  [max_strain]
    type = InitializedSymR2
    values = '0.1 -0.05 -0.05'
    nbatch = 20
  []
  [strains]
    type = LinSpaceTensor
    end = max_strain
    steps = 100
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
    type = SymR2ForceRate
    force = 'E'
  []
  [stress_rate]
    type = LinearElasticity
    youngs_modulus = 1e5
    poisson_ratio = 0.3
    rate_form = true
    strain = 'forces/E'
  []
  [integrate]
    type = SymR2ForwardEulerTimeIntegration
    variable = 'S'
  []
  [model]
    type = ComposedModel
    models = 'force_rate stress_rate integrate'
  []
[]
