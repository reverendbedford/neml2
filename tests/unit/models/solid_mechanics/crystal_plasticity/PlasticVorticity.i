[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '(10,3)'
    output_skewr2_names = 'state/internal/plastic_vorticity'
    output_skewr2_values = 'wp'
    input_rot_names = 'state/orientation'
    input_rot_values = 'R'
    input_batch_tensor_names = 'state/internal/slip_rates'
    input_batch_tensor_values = 'gamma'
    derivatives_rel_tol = 0
    derivatives_abs_tol = 5e-6
    second_derivatives_rel_tol = 0
    second_derivatives_abs_tol = 5e-6
    check_AD_first_derivatives = false
    check_AD_second_derivatives = false
    check_AD_derivatives = false
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
    values = '0.00499066 -0.0249533 0.03493462'
  []
  [gamma]
    type = LinspaceTensor
    start = -0.1
    end = 0.2
    nstep = 12
    dim = 0
    batch_dim = 0
    batch_expand = '(10 3)'
  []
  [wp]
    type = FillWR2
    values = '-0.09829713 -0.01517324 0.09810889'
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
  [euler_rodrigues]
    type = RotationMatrix
    from = 'state/orientation'
    to = 'state/orientation_matrix'
  []
  [vorticity]
    type = PlasticVorticity
  []
  [model]
    type = ComposedModel
    models = 'euler_rodrigues vorticity'
  []
[]
