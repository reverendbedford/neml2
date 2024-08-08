[Solvers]
  [newton]
    type = Newton
  []
[]

[Models]
  [vonmises]
    type = SR2Invariant
    tensor = 'forces/S'
    invariant = 'forces/s'
    invariant_type = 'VONMISES'
  []
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 500
  []
  [yield]
    type = YieldFunction
    yield_stress = 10
    effective_stress = 'forces/s'
    isotropic_hardening = 'state/internal/k'
  []
  [n]
    type = ArrheniusParameter
    reference_value = 2
    activation_energy = -5e3
    ideal_gas_constant = 8.314
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 'n'
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'state/internal/fp'
    from = 'state/internal/k forces/S'
    to = 'state/internal/Nk state/internal/NM'
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/ep'
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/Ep'
  []
  [surface]
    type = ComposedModel
    models = "isoharden yield flow_rate normality
              eprate Eprate
              integrate_ep integrate_Ep"
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = 'surface'
    solver = 'newton'
  []
  [stress_update]
    type = ComposedModel
    models = 'vonmises return_map'
  []
  [elastic_strain]
    type = LinearIsotropicElasticity
    youngs_modulus = 3e4
    poisson_ratio = 0.3
    compliance = true
    stress = 'forces/S'
  []
  [total_strain]
    type = SR2LinearCombination
    to_var = 'state/E'
    from_var = 'state/internal/Ee state/internal/Ep'
  []
  [model]
    type = ComposedModel
    models = 'stress_update elastic_strain total_strain'
    additional_outputs = 'state/internal/ep state/internal/Ep'
  []
[]
