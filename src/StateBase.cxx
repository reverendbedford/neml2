#include "StateBase.h"

StateBase::StateBase(const torch::Tensor & tensor)
  : StandardBatchedLabeledTensor(tensor)
{
}

TorchSize
StateBase::batch_size() const
{
  return sizes()[0];
}
