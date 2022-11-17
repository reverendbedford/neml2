#include "tensors/Scalar.h"

Scalar::Scalar(double init, TorchSize batch_size)
  : FixedDimTensor<1, 1>(torch::tensor(init, TorchDefaults), batch_size)
{
}
