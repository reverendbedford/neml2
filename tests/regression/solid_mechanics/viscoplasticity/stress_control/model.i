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
  [sxx]
    type = FullScalar
    batch_shape = '(20)'
    value = 120
  []
  [syy]
    type = FullScalar
    batch_shape = '(20)'
    value = 0
  []
  [szz]
    type = FullScalar
    batch_shape = '(20)'
    value = 0
  []
  [max_stress]
    type = FillSR2
    values = 'sxx syy szz'
  []
  [stresses]
    type = LinspaceSR2
    start = 0
    end = max_stress
    nstep = 100
  []
[]

[Drivers]
  [driver]
    type = SolidMechanicsDriver
    model = 'model'
    times = 'times'
    control = 'STRESS'
    prescribed_stresses = 'stresses'
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
    type = NewtonNonlinearSolver
  []
[]

[Models]
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
  [elastic_strain]
    type = ElasticStrain
    total_strain = 'state/E'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    youngs_modulus = 1e5
    poisson_ratio = 0.3
  []
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/M'
    invariant = 'state/internal/sm'
  []
  [yield]
    type = YieldFunction
    yield_stress = 5
    isotropic_hardening = 'state/internal/k'
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'state/internal/fp'
    from = 'state/internal/M state/internal/k'
    to = 'state/internal/NM state/internal/Nk'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [Srate]
    type = SR2ForceRate
    force = 'S'
  []
  [Eerate]
    type = LinearIsotropicElasticity
    stress = 'forces/S'
    youngs_modulus = 1e5
    poisson_ratio = 0.3
    rate_form = true
    compliance = true
  []
  [Erate]
    type = TotalStrain
    rate_form = true
  []

  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'internal/ep'
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'internal/Ep'
  []
  [integrate_E]
    type = SR2BackwardEulerTimeIntegration
    variable = 'E'
  []
  [implicit_rate]
    type = ComposedModel
    models = "isoharden elastic_strain elasticity mandel_stress vonmises yield
              normality flow_rate eprate Eprate Srate Eerate Erate
              integrate_ep integrate_Ep integrate_E"
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
[]
