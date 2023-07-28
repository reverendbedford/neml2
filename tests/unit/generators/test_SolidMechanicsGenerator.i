[Solvers]
  [newton]
    type = NewtonNonlinearSolver
  []
[]

[Predictors]
  [simple]
    type = LinearExtrapolationPredictor
  []
[]

[SolidMechanics]
  [Elasticity]
    [linear]
      type = LinearElasticity
      E = 1e5
      nu = 0.3
    []
  []
  [Viscoplasticity]
    solver = 'newton'
    predictor = 'simple'
    [FlowRate]
      [perzyna]
        type = PerzynaPlasticFlowRate
        eta = 100
        n = 2
      []
    []
  []
[]
