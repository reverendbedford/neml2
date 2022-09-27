#pragma once

#include "StandardBatchedTensor.h"

/// A batch of scalars stored as (nbatch,1)
// This is actually a non-trivial decision, I guess we'll go with
// scalars have shape (1,)
using BatchedScalarBase = StandardBatchedTensor<1>;

/// A batch of scalars stored as a (nbatch,1)
class BatchedScalar : public BatchedScalarBase
{
public:
  BatchedScalar(TorchSize nbatch);
  BatchedScalar(const torch::Tensor & tensor);
};
