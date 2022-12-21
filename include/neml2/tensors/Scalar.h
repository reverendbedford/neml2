#pragma once

#include "neml2/tensors/FixedDimTensor.h"

namespace neml2
{
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

  /// Fill with zeros
  static Scalar zeros(TorchSize batch_size);

  /// Negation
  Scalar operator-() const;

  /// Exponentiation
  Scalar pow(Scalar n) const;

  /// The derivative of a Scalar with respect to itself
  [[nodiscard]] static Scalar identity_map() { return 1; }
};

// We would like to have exact match for the basic operators to avoid ambiguity, and also to keep
// the return type as one of our supported primitive tensor types. All we need to do is to forward
// the calls to torch :)
/// @{
Scalar operator+(const Scalar & a, const Scalar & b);
BatchTensor<1> operator+(const BatchTensor<1> & a, const Scalar & b);
BatchTensor<1> operator+(const Scalar & a, const BatchTensor<1> & b);

Scalar operator-(const Scalar & a, const Scalar & b);
BatchTensor<1> operator-(const BatchTensor<1> & a, const Scalar & b);
BatchTensor<1> operator-(const Scalar & a, const BatchTensor<1> & b);

Scalar operator*(const Scalar & a, const Scalar & b);
BatchTensor<1> operator*(const BatchTensor<1> & a, const Scalar & b);
BatchTensor<1> operator*(const Scalar & a, const BatchTensor<1> & b);

Scalar operator/(const Scalar & a, const Scalar & b);
BatchTensor<1> operator/(const BatchTensor<1> & a, const Scalar & b);
BatchTensor<1> operator/(const Scalar & a, const BatchTensor<1> & b);
/// @}

/// Macaulay bracket and its derivative
/// @{
Scalar macaulay(const Scalar & a, const Scalar & a0);
Scalar dmacaulay(const Scalar & a, const Scalar & a0);
/// @}

/// Exponential function
Scalar exp(const Scalar & a);

} // namespace neml2
