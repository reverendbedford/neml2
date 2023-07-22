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
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 1.2
  []
  [elastic_strain]
    type = ElasticStrain
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
    yield_stress = 1000
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
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'internal/ep'
  []
  [integrate_Ep]
    type = SymR2BackwardEulerTimeIntegration
    variable = 'internal/Ep'
  []
  [consistency]
    type = RateIndependentPlasticFlowConstraint
  []
  [surface]
    type = ComposedModel
    models = "isoharden elastic_strain elasticity
              mandel_stress vonmises
              yield normality eprate Eprate
              consistency integrate_ep integrate_Ep"
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = 'surface'
    solver = 'newton'
    additional_outputs = 'state/internal/Ep state/internal/ep'
  []
  [model]
    type = ComposedModel
    models = 'return_map elastic_strain elasticity'
  []
[]
