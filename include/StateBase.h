#pragma once

#include "StandardBatchedLabeledTensor.h"

/// Common class for defining state vectors and state derivative matrices
class StateBase : public StandardBatchedLabeledTensor
{
public:
  StateBase(const torch::Tensor & tensor);

  /// Return the batch size
  virtual TorchSize batch_size() const;
};
