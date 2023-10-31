nb = 20
nt = 100

[Tensors]
  [end_time]
    type = LogspaceScalar
    start = -1
    end = 5
    nstep = ${nb}
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = ${nt}
  []
  [exx]
    type = FullScalar
    batch_shape = '(${nb})'
    value = 0.1
  []
  [eyy]
    type = FullScalar
    batch_shape = '(${nb})'
    value = -0.05
  []
  [ezz]
    type = FullScalar
    batch_shape = '(${nb})'
    value = -0.05
  []
  [max_strain]
    type = FillSR2
    values = 'exx eyy ezz'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = ${nt}
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
  ###############################################################################
  # Use the trial state to precalculate invariant flow directions 
  # prior to radial return
  ###############################################################################
  [trial_elastic_strain]
    type = ElasticStrain
    plastic_strain = 'old_state/internal/Ep'
  []
  [cauchy_stress]
    type = LinearIsotropicElasticity
    youngs_modulus = 1e5
    poisson_ratio = 0.3
  []
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [trial_isoharden]
    type = LinearIsotropicHardening
    equivalent_plastic_strain = 'old_state/internal/ep'
    hardening_modulus = 1000
  []
  [trial_kinharden]
    type = LinearKinematicHardening
    kinematic_plastic_strain = 'old_state/internal/Kp'
    hardening_modulus = 1000
  []
  [overstress]
    type = OverStress
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/O'
    invariant = 'state/internal/sm'
  []
  [yield]
    type = YieldFunction
    yield_stress = 1000
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
    from = 'state/internal/M state/internal/k state/internal/X'
    to = 'forces/NM forces/Nk forces/NX'
  []
  [trial_state]
    type = ComposedModel
    models = "trial_elastic_strain cauchy_stress mandel_stress
              trial_isoharden trial_kinharden normality"
  []
  ###############################################################################
  # The actual radial return:
  # Since the flow directions are invariant, we only need to solve 
  # the consistency condition.
  ###############################################################################
  [plastic_strain_rate]
    type = AssociativePlasticFlow
    flow_direction = 'forces/NM'
  []
  [plastic_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'internal/Ep'
  []
  [elastic_strain]
    type = ElasticStrain
  []
  [equivalent_plastic_strain_rate]
    type = AssociativeIsotropicPlasticHardening
    isotropic_hardening_direction = 'forces/Nk'
  []
  [equivalent_plastic_strain]
    type = ScalarForwardEulerTimeIntegration
    variable = 'internal/ep'
  []
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
  [kinematic_plastic_strain_rate]
    type = AssociativeKinematicPlasticHardening
    kinematic_hardening_direction = 'forces/NX'
  []
  [kinematic_plastic_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'internal/Kp'
  []
  [kinharden]
    type = LinearKinematicHardening
    hardening_modulus = 1000
  []
  [consistency]
    type = RateIndependentPlasticFlowConstraint
  []
  [surface]
    type = ComposedModel
    models = "plastic_strain_rate plastic_strain elastic_strain cauchy_stress mandel_stress 
              kinematic_plastic_strain_rate kinematic_plastic_strain kinharden 
              equivalent_plastic_strain_rate equivalent_plastic_strain isoharden 
              overstress vonmises yield consistency"
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = 'surface'
    solver = 'newton'
  []
  [model0]
    type = ComposedModel
    models = "trial_state return_map 
              plastic_strain_rate plastic_strain
              equivalent_plastic_strain_rate equivalent_plastic_strain
              kinematic_plastic_strain_rate kinematic_plastic_strain"
    additional_outputs = 'state/internal/Ep state/internal/ep state/internal/Kp'
  []
  [model]
    type = ComposedModel
    models = 'model0 elastic_strain cauchy_stress'
  []
[]
