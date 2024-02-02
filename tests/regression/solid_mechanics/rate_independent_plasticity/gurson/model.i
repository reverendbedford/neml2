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
    value = -0.03
  []
  [ezz]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.01
  []
  [eyz]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.01
  []
  [exz]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.02
  []
  [exy]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.015
  []
  [max_strain]
    type = FillSR2
    values = 'exx eyy ezz eyz exz exy'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = 100
  []
  [f0]
    type = Scalar
    values = '0.01'
  []
[]

[Drivers]
  [driver]
    type = SolidMechanicsDriver
    model = 'model'
    times = 'times'
    prescribed_strains = 'strains'
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
    type = NewtonNonlinearSolver
    linesearch = true
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
    invariant = 'state/internal/sp'
  []
  [yield]
    type = GTNYieldFunction
    yield_stress = 60.0
    q1 = 1.25
    q2 = 1.0
    q3 = 1.57
    isotropic_hardening = 'state/internal/k'
  []
  [flow]
    type = ComposedModel
    models = 'j2 i1 yield'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'state/internal/fp'
    from = 'state/internal/M state/internal/k'
    to = 'state/internal/NM state/internal/Nk'
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'internal/Ep'
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'internal/ep'
  []
  [consistency]
    type = RateIndependentPlasticFlowConstraint
  []
  [voidrate]
    type = GursonCavitation
  []
  [integrate_voidrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'internal/f'
  []
  [surface]
    type = ComposedModel
    models = "isoharden elastic_strain elasticity
              mandel_stress j2 i1
              yield normality Eprate voidrate
              consistency integrate_Ep integrate_voidrate eprate integrate_ep"
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = 'surface'
    solver = 'newton'
    additional_outputs = 'state/internal/Ep'
  []
  [model]
    type = ComposedModel
    models = 'return_map elastic_strain elasticity'
  []
[]
