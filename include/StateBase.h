#pragma once

#include "LabeledTensor.h"

/// Common class for defining state vectors and state derivative matrices
class StateBase : public LabeledTensor<1>
{
public:
  /// Forward all the constructors
  using LabeledTensor<1>::LabeledTensor;

  /// Return the batch size
  TorchSize batch_size() const;
};
