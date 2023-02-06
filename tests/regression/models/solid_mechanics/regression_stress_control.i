[Solvers]
  [newton]
    type = NewtonNonlinearSolver
  []
[]

[Models]
  [load]
    type = SymR2IdentityMap
    from_var = 'forces cauchy_stress'
    to_var = 'state cauchy_stress'
  []
  [M]
    type = IsotropicMandelStress
  []
  [gamma]
    type = LinearIsotropicHardening
    K = 1000
  []
  [j2]
    type = J2StressMeasure
  []
  [f]
    type = IsotropicHardeningYieldFunction
    stress_measure = j2
    yield_stress = 5
  []
  [gammarate]
    type = PerzynaPlasticFlowRate
    eta = 100
    n = 2
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
    yield_function = f
  []
  [rate]
    type = ComposedModel
    models = 'load M gamma f gammarate eprate'
  []
  [surface]
    type = ImplicitTimeIntegration
    rate = rate
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = surface
    solver = newton
    additional_outputs = 'state internal_state equivalent_plastic_strain'
  []
  [Np]
    type = AssociativePlasticFlowDirection
    yield_function = f
  []
  [Eprate]
    type = PlasticStrainRate
  []
  [Ep]
    type = SymR2TimeIntegration
    variable = plastic_strain
    additional_outputs = 'state plastic_strain'
  []
  [Ee]
    type = ElasticStrainFromCauchyStress
    E = 1e5
    nu = 0.3
  []
  [strain]
    type = TotalStrain
  []
  [model]
    type = ComposedModel
    models = 'load return_map gamma M f Np gammarate Eprate Ep Ee strain'
  []
[]
