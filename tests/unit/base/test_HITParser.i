[Solvers]
  [newton]
    type = NewtonNonlinearSolver
  []
[]

[Models]
  [foo]
    type = ScalarSumModel
    from_var = "state A; state substate B"
    to_var = "state outsub C"
  []
  [bar]
    type = ScalarSumModel
    from_var = "state A; state substate B"
    to_var = "state outsub C"
  []
[]
