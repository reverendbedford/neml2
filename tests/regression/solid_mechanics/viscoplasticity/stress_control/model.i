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
    values = '120 0 0'
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
    K = 1000
  []
  [elastic_strain]
    type = ElasticStrain
    total_strain = 'state/E'
  []
  [elasticity]
    type = LinearElasticity
    E = 1e5
    nu = 0.3
  []
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [vonmises]
    type = SymR2Invariant
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
    eta = 100
    n = 2
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [Srate]
    type = SymR2ForceRate
    force = 'S'
  []
  [Eerate]
    type = LinearElasticity
    stress = 'forces/S'
    E = 1e5
    nu = 0.3
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
    type = SymR2BackwardEulerTimeIntegration
    variable = 'internal/Ep'
  []
  [integrate_E]
    type = SymR2BackwardEulerTimeIntegration
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
