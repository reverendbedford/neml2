#pragma once

#include "StandardBatchedTensor.h"
#include "SymSymR4.h"

/// Using Mandel notation this is a (nbatch,6,6)
using BatchedSymSymR4Base = StandardBatchedTensor<6, 6>;

/// A Mandel notation rank 4 with minor symmetry tensor stored as (nbatch,6,6)
class BatchedSymSymR4 : public BatchedSymSymR4Base
{
public:
  /// Construct blank given the batch size
  BatchedSymSymR4(TorchSize nbatch);
  /// Construct from another generic tensor
  BatchedSymSymR4(const torch::Tensor & tensor);
};
