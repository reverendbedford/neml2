nbatch = 20

[Tensors]
  [end_time]
    type = LogspaceScalar
    start = 3
    end = 3
    nstep = ${nbatch}
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = 100
  []
  [start_temperature]
    type = LinspaceScalar
    start = 300
    end = 300
    nstep = ${nbatch}
  []
  [end_temperature]
    type = LinspaceScalar
    start = 1800
    end = 1800
    nstep = ${nbatch}
  []
  [temperatures]
    type = LinspaceScalar
    start = start_temperature
    end = end_temperature
    nstep = 100
  []
  [exx]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = 0
  []
  [eyy]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = 0
  []
  [ezz]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = 0
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
  [f0]
    type = Scalar
    values = '0.36'
  []
[]

[Drivers]
  [driver]
    type = SolidMechanicsDriver
    model = 'model'
    times = 'times'
    prescribed_strains = 'strains'
    prescribed_temperatures = 'temperatures'
    ic_scalar_names = 'state/internal/f'
    ic_scalar_values = 'f0'
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
  []
[]

[Models]
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 5
    saturation_rate = 1.2
  []
  [elastic_strain]
    type = ElasticStrain
  []
  [elasticity]
    type = LinearIsotropicElasticity
    youngs_modulus = 30000
    poisson_ratio = 0.3
  []
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [j2]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/M'
    invariant = 'state/internal/se'
  []
  [i1]
    type = SR2Invariant
    invariant_type = 'I1'
    tensor = 'state/internal/M'
    invariant = 'state/internal/sh'
  []
  [ss]
    type = OlevskySinteringStress
    surface_tension = 5e-3
    particle_radius = 3e-5
  []
  [sp]
    type = ScalarSumModel
    to_var = 'state/internal/sp'
    from_var = 'state/internal/sh state/internal/ss'
    coefficients = '1 -1'
  []
  [q1]
    type = ArrheniusParameter
    temperature = 'forces/T'
    reference_value = 500
    activation_energy = 5e4
    ideal_gas_constant = 8.314
  []
  [yield]
    type = GTNYieldFunction
    yield_stress = 60.0
    q1 = 'q1'
    q2 = 1.0
    q3 = 1.57
    isotropic_hardening = 'state/internal/k'
  []
  [flow]
    type = ComposedModel
    models = 'q1 j2 i1 ss sp yield'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 1000
    exponent = 2
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'state/internal/fp'
    from = 'state/internal/M state/internal/k state/internal/f'
    to = 'state/internal/NM state/internal/Nk state/internal/Nf'
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [voidrate]
    type = AssociativeCavitation
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'internal/Ep'
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'internal/ep'
  []
  [integrate_void]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'internal/f'
  []
  [surface]
    type = ComposedModel
    models = "isoharden elastic_strain elasticity
              mandel_stress flow flow_rate
              normality
              Eprate eprate voidrate
              integrate_Ep integrate_ep integrate_void"
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = 'surface'
    solver = 'newton'
  []
  [model]
    type = ComposedModel
    models = 'return_map elastic_strain elasticity'
    additional_outputs = 'state/internal/Ep state/internal/ep state/internal/f'
  []
[]
