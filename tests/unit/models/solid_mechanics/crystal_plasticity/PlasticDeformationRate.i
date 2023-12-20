[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '(10,3)'
    output_symr2_names = 'state/internal/plastic_deformation_rate'
    output_symr2_values = 'dp'
    input_rot_names = 'state/orientation'
    input_rot_values = 'R'
    input_batch_tensor_names = 'state/internal/slip_rates'
    input_batch_tensor_values = 'gamma'
    derivatives_rel_tol = 0
    derivatives_abs_tol = 5e-6
    second_derivatives_rel_tol = 0
    second_derivatives_abs_tol = 5e-6
  []
[]

[Tensors]
  [a]
    type = Scalar
    values = '1.2'
  []
  [sdirs]
    type = FillMillerIndex
    values = '1 1 0'
  []
  [splanes]
    type = FillMillerIndex
    values = '1 1 1'
  []
  [R]
    type = FillRot
    values = '0.01 -0.05 0.07'
  []
  [gamma]
    type = LinspaceBatchTensor
    start = -0.1
    end = 0.2
    nstep = 12
    dim = 0
    batch_dim = 0
    batch_expand = '(10 3)'
  []
  [dp]
    type = FillSR2
    values = '0.0546068 -0.0421977 -0.0124091 0.123251 -0.0935403 0.0278809'
  []
[]

[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = "a"
    slip_directions = "sdirs"
    slip_planes = "splanes"
  []
[]

[Models]
  [model]
    type = PlasticDeformationRate
  []
[]
