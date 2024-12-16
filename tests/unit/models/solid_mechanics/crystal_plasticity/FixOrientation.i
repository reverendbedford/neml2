[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Rot_names = 'state/orientation'
    input_Rot_values = 'R_in'
    output_Rot_names = 'state/output_orientation'
    output_Rot_values = 'R_out'
  []
[]

[Tensors]
  [R_in]
    type = FillRot
    values = '1.0 -0.1 -0.05'
  []
  [R_out]
    type = FillRot
    values = '-0.98765432 0.09876543 0.04938272'
  []
[]

[Models]
  [model]
    type = FixOrientation
    input_orientation = 'state/orientation'
    output_orientation = 'state/output_orientation'
  []
[]
