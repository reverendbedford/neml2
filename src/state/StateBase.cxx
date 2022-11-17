#include "state/StateBase.h"

TorchSize
StateBase::batch_size() const
{
  return tensor().batch_sizes()[0];
}
