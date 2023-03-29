[Solvers]
  [newton]
    type = NewtonNonlinearSolver
  []
[]

[Models]
  [Ee]
    type = ElasticStrain
  []
  [S]
    type = CauchyStressFromElasticStrain
    E = 1e5
    nu = 0.3
  []
  [M]
    type = IsotropicMandelStress
  []
  [j2]
    type = J2StressMeasure
  []
  [f]
    type = PerfectlyPlasticYieldFunction
    stress_measure = j2
    yield_stress = 1000
  []
  [Np]
    type = AssociativePlasticFlowDirection
    yield_function = f
  []
  [Eprate]
    type = PlasticStrainRate
  []
  [integrate_Ep]
    type = SymR2BackwardEulerTimeIntegration
    rate_variable = plastic_strain_rate
    variable = plastic_strain
  []
  [consistency]
    type = RateIndependentPlasticFlowConstraint
  []
  [surface]
    type = ComposedModel
    models = 'Ee S M Np Eprate integrate_Ep f consistency'
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = surface
    solver = newton
    additional_outputs = 'state plastic_strain'
  []
  [model]
    type = ComposedModel
    models = 'return_map Ee S'
  []
[]
