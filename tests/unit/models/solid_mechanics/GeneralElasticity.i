[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '(10)'
    input_symr2_names = 'state/internal/Ee'
    input_symr2_values = 'Ee'
    input_rot_names = 'state/orientation'
    input_rot_values = 'R'
    output_symr2_names = 'state/S'
    output_symr2_values = 'S'
    derivatives_abs_tol = 1e-6
    derivatives_rel_tol = 1e-4
  []
[]

[Tensors]
  [Ee]
    type = FillSR2
    values = '0.09 0.04 -0.02'
  []
  [R]
    type = FillRot
    values = '0.13991834 0.18234513 0.85043991'
  []
  [S]
    type = FillSR2
    values = '-0.2731859829 -0.3965257479 0.7095832817 -0.0316765403 -0.4720823316 0.3265491460'
  []
  [C]
    type = SSR4
    values = " 1  2  3  4  5  6
               7  8  9 10 11 12
              13 14 15 16 17 18
              19 20 21 22 23 24
              25 26 27 28 29 30
              31 32 33 34 35 36"
    batch_shape = '(10)'
  []
[]

[Models]
  [model]
    type = GeneralElasticity
    elastic_stiffness_tensor = 'C'
  []
[]
