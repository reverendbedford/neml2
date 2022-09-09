#pragma once

#include "StandardUnbatchedTensor.h"

/// A single symmetric rank 2 tensor stored in Mandel notation as (6,)
using SymR2Base = StandardUnbatchedTensor<6>;

/// A single symmetric rank 2 tensor stored in Mandel notation with size (6,)
class SymR2: public SymR2Base {
 public:
  SymR2();
  SymR2(const torch::Tensor & tensor);
};
