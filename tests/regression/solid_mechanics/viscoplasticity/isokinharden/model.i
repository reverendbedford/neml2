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
  [kinharden]
    type = LinearKinematicHardening
    H = 1000
  []
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [overstress]
    type = OverStress
  []
  [vonmises]
    type = SymR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/O'
    invariant = 'state/internal/sm'
  []
  [yield]
    type = YieldFunction
    yield_stress = 5
    isotropic_hardening = 'state/internal/k'
  []
  [flow]
    type = ComposedModel
    models = 'overstress vonmises yield'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'state/internal/fp'
    from = 'state/internal/M state/internal/k state/internal/X'
    to = 'state/internal/NM state/internal/Nk state/internal/NX'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    eta = 100
    n = 2
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [Kprate]
    type = AssociativeKinematicPlasticHardening
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [Erate]
    type = SymR2ForceRate
    force = 'E'
  []
  [Eerate]
    type = ElasticStrain
    rate_form = true
  []
  [elasticity]
    type = LinearElasticity
    E = 1e5
    nu = 0.3
    rate_form = true
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'internal/ep'
  []
  [integrate_Kp]
    type = SymR2BackwardEulerTimeIntegration
    variable = 'internal/Kp'
  []
  [integrate_stress]
    type = SymR2BackwardEulerTimeIntegration
    variable = 'S'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'isoharden kinharden mandel_stress overstress vonmises yield normality flow_rate eprate Eprate Kprate Erate Eerate elasticity integrate_stress integrate_ep integrate_Kp'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
[]
