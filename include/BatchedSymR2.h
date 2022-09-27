#pragma once

#include "StandardBatchedTensor.h"

/// Using Mandel notation this is a (nbatch,6)
using BatchedSymR2Base = StandardBatchedTensor<6>;

/// A Mandel notation symmetric rank 2 tensor stored as (nbatch,6)
class BatchedSymR2 : public BatchedSymR2Base
{
public:
  BatchedSymR2(TorchSize nbatch);
  BatchedSymR2(const torch::Tensor & tensor);
};
