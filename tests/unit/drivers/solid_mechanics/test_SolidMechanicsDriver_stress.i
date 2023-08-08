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
  [max_stress]
    type = InitializedSymR2
    values = '0.1 -0.05 -0.05'
    nbatch = 20
  []
  [stresses]
    type = LinSpaceTensor
    end = max_stress
    steps = 100
  []
[]

[Drivers]
  [driver]
    type = SolidMechanicsDriver
    model = 'model'
    times = 'times'
    prescribed_stresses = 'stresses'
    control = 'STRESS'
    save_as = 'unit/drivers/solid_mechanics/test_SolidMechanicsDriver_stress.pt'
  []
[]

[Models]
  [force_rate]
    type = SymR2ForceRate
    force = 'S'
  []
  [strain_rate]
    type = LinearIsotropicElasticity
    youngs_modulus = 1e5
    poisson_ratio = 0.3
    rate_form = true
    compliance = true
    stress = 'forces/S'
    strain = 'state/E'
  []
  [integrate]
    type = SymR2ForwardEulerTimeIntegration
    variable = 'E'
  []
  [model]
    type = ComposedModel
    models = 'force_rate strain_rate integrate'
  []
[]
