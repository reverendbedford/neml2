#pragma once

#include "FixedDimTensor.h"

/// A single scalar stored as a (B,) tensor
///  Consider a typedef on the scalar type...
//
/// IMPORTANT: The base size of a Scalar is _empty_ instead of 1. This choice is made such that NEML2
/// is more conformant with libtorch.
class Scalar : public FixedDimTensor<1, 1>
{
public:
  /// Forward all the constructors
  using FixedDimTensor<1, 1>::FixedDimTensor;

  /// A convenient conversion to allow people to do
  /// ~~~~~~~~~~~~~~~~~~~~cpp
  /// Scalar a = 1.0;
  /// Scalar b(5.6);
  /// Scalar c(2.3, 1000);
  /// ~~~~~~~~~~~~~~~~~~~~
  Scalar(double init, TorchSize batch_size = 1);
};
