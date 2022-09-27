#include "BatchedSymR2.h"

BatchedSymR2::BatchedSymR2(TorchSize nbatch)
  : BatchedSymR2Base(nbatch)
{
}

BatchedSymR2::BatchedSymR2(const torch::Tensor & tensor)
  : BatchedSymR2Base(tensor)
{
}
