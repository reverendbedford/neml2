#pragma once

#include "StandardUnbatchedTensor.h"

/// A scalar stored as a (1,)
using ScalarBase = StandardUnbatchedTensor<1>;

/// A single scalar stored as a (1,) tensor
class Scalar: public ScalarBase {
 public:
  Scalar();
  Scalar(const torch::Tensor & tensor);
};
