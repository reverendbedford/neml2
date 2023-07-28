[Tensors]
  [times]
    type = VTestTimeSeries
    vtest = 'isolinear_multiaxial.vtest'
    variable = 'time'
    variable_type = 'SCALAR'
  []
  [strains]
    type = VTestTimeSeries
    vtest = 'isolinear_multiaxial.vtest'
    variable = 'strain'
    variable_type = 'SYMR2'
  []
  [stresses]
    type = VTestTimeSeries
    vtest = 'isolinear_multiaxial.vtest'
    variable = 'stress'
    variable_type = 'SYMR2'
  []
[]

[Drivers]
  [driver]
    type = SolidMechanicsDriver
    model = 'model'
    times = 'times'
    prescribed_strains = 'strains'
  []
  [verification]
    type = VTestVerification
    driver = 'driver'
    variables = 'output.state/S'
    references = 'stresses'
    rtol = 1e-5
    atol = 1e-8
  []
[]

[Solvers]
  [newton]
    type = NewtonNonlinearSolver
  []
[]

[Models]
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [vonmises]
    type = SymR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/M'
    invariant = 'state/internal/sm'
  []
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 5500
  []
  [yield]
    type = YieldFunction
    yield_stress = 10
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
    reference_stress = 500
    exponent = 5
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
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
    youngs_modulus = 124000
    poisson_ratio = 0.32
    rate_form = true
  []
  [integrate_stress]
    type = SymR2BackwardEulerTimeIntegration
    variable = 'S'
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'internal/ep'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'mandel_stress vonmises isoharden yield normality flow_rate Eprate eprate Erate Eerate elasticity integrate_stress integrate_ep'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
[]
