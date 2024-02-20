[Tensors]
  [times]
    type = VTestTimeSeries
    vtest = 'perfect.vtest'
    variable = 'time'
    variable_type = 'SCALAR'
  []
  [strains]
    type = VTestTimeSeries
    vtest = 'perfect.vtest'
    variable = 'strain'
    variable_type = 'SYMR2'
  []
  [stresses]
    type = VTestTimeSeries
    vtest = 'perfect.vtest'
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
    type = Newton
  []
[]

[Models]
  [elastic_strain]
    type = ElasticStrain
  []
  [elasticity]
    type = LinearIsotropicElasticity
    youngs_modulus = 120000
    poisson_ratio = 0.3
  []
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/M'
    invariant = 'state/internal/s'
  []
  [yield]
    type = YieldFunction
    yield_stress = 100
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield'
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
  [surface]
    type = ComposedModel
    models = "elastic_strain elasticity
              mandel_stress vonmises
              yield normality Eprate
              consistency integrate_Ep"
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
