#pragma once

#include "StandardUnbatchedTensor.h"

#include "SymR2.h"

/// A rank 4 tensor with both minor symmetries
using SymSymR4Base = StandardUnbatchedTensor<6,6>;

/// A rank 4 tensor with minor symmetry stored in Mandel notation as a (6,6)
class SymSymR4: public SymSymR4Base {
 public:
  SymSymR4();
  SymSymR4(const torch::Tensor & tensor);
};
