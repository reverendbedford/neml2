[Solvers]
  [newton]
    type = Newton
  []
[]

[Models]
  [isoharden1]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 150
    isotropic_hardening = 'state/internal/k1'
  []
  [isoharden2]
    type = VoceIsotropicHardening
    saturated_hardening = -80
    saturation_rate = 80
    isotropic_hardening = 'state/internal/k2'
  []
  [isoharden]
    type = ScalarLinearCombination
    from_var = 'state/internal/k1 state/internal/k2'
    to_var = 'state/internal/k'
  []
  [kinharden]
    type = SR2LinearCombination
    from_var = 'state/internal/X1 state/internal/X2'
    to_var = 'state/internal/X'
  []
  [elastic_strain]
    type = SR2LinearCombination
    from_var = 'forces/E state/internal/Ep'
    to_var = 'state/internal/Ee'
    coefficients = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    youngs_modulus = 2e4
    poisson_ratio = 0.3
  []
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [overstress]
    type = SR2LinearCombination
    to_var = 'state/internal/O'
    from_var = 'state/internal/M state/internal/X'
    coefficients = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/O'
    invariant = 'state/internal/s'
  []
  [yield]
    type = YieldFunction
    yield_stress = 50
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
    from = 'state/internal/M state/internal/k'
    to = 'state/internal/NM state/internal/Nk'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [X1rate]
    type = ChabochePlasticHardening
    back_stress = 'state/internal/X1'
    C = 1000
    g = 0
    A = 0
    a = 3
  []
  [X2rate]
    type = ChabochePlasticHardening
    back_stress = 'state/internal/X2'
    C = 1000
    g = 0
    A = 0
    a = 3
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/ep'
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/Ep'
  []
  [integrate_X1]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/X1'
  []
  [integrate_X2]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/X2'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'isoharden1 isoharden2 isoharden
              kinharden
              elastic_strain elasticity mandel_stress
              overstress vonmises yield
              normality
              flow_rate eprate Eprate X1rate X2rate
              integrate_ep integrate_Ep integrate_X1 integrate_X2'
  []
  [model0]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
  [model]
    type = ComposedModel
    models = 'model0 elastic_strain elasticity mandel_stress kinharden overstress vonmises'
    additional_outputs = 'state/internal/Ep state/internal/X1 state/internal/X2 state/S'
  []
[]
