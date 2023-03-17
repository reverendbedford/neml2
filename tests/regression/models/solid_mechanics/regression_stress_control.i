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
  [h]
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
  [hrate]
    type = PerzynaPlasticFlowRate
    eta = 100
    n = 2
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
    yield_function = f
  []
  [integrate_ep]
    type = ScalarImplicitTimeIntegration
    rate_variable = 'internal_state equivalent_plastic_strain_rate'
    variable = 'internal_state equivalent_plastic_strain'
  []
  [surface]
    type = ComposedModel
    models = 'load M h f hrate eprate integrate_ep'
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
    models = 'load return_map h M f Np hrate Eprate Ep Ee strain'
  []
[]
