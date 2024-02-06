[Tensors]
  [times]
    type = VTestTimeSeries
    vtest = 'gurson.vtest'
    variable = 'time'
    variable_type = 'SCALAR'
  []
  [strains]
    type = VTestTimeSeries
    vtest = 'gurson.vtest'
    variable = 'strain'
    variable_type = 'SYMR2'
  []
  [stresses]
    type = VTestTimeSeries
    vtest = 'gurson.vtest'
    variable = 'stress'
    variable_type = 'SYMR2'
  []
  [f0]
    type = Scalar
    values = '0.002'
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
  []
  [verification]
    type = VTestVerification
    driver = 'driver'
    variables = 'output.state/S'
    references = 'stresses'
    rtol = 1e-2
    atol = 1e-5
  []
[]

[Solvers]
  [newton]
    type = NewtonNonlinearSolver
    linesearch = true
  []
[]

[Models]
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
  []
  [flow]
    type = ComposedModel
    models = 'j2 i1 yield'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'state/internal/fp'
    from = 'state/internal/M'
    to = 'state/internal/NM'
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'internal/Ep'
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
    models = "elastic_strain elasticity
              mandel_stress j2 i1
              yield normality Eprate voidrate
              consistency integrate_Ep integrate_voidrate"
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = 'surface'
    solver = 'newton'
  []
  [model]
    type = ComposedModel
    models = 'return_map elastic_strain elasticity'
    additional_outputs = 'state/internal/Ep'
  []
[]
