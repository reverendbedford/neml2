#pragma once

#include "tensors/Scalar.h"
#include "utils.h"

/// Forward decl
class SymSymR4;

/// A single symmetric rank 2 tensor stored in Mandel notation stored as a (B, 6,) tensor
class SymR2 : public FixedDimTensor<1, 6>
{
public:
  using FixedDimTensor<1, 6>::FixedDimTensor;

  /// Named constructors
  /// @{
  /// Make zero with some batch size
  static SymR2 zeros();
  /// Fill the diagonals with a11 = a22 = a33 = a
  static SymR2 init(const Scalar & a);
  /// Fill the diagonals with a11, a22, a33
  static SymR2 init(const Scalar & a11, const Scalar & a22, const Scalar & a33);
  /// Fill all entries
  static SymR2 init(const Scalar & a11,
                    const Scalar & a22,
                    const Scalar & a33,
                    const Scalar & a23,
                    const Scalar & a13,
                    const Scalar & a12);
  /// Identity
  static SymR2 identity();
  /// @}

  /// Accessor
  Scalar operator()(TorchSize i, TorchSize j) const;

  SymR2 operator-(const SymR2 & other) const;

  /// Trace
  Scalar tr() const;

  /// Volumetric part of the tensor
  SymR2 vol() const;

  /// Deviatoric part of the tensor
  SymR2 dev() const;

  /// Determinant
  Scalar det() const;

  /// Double contraction ij,ij
  Scalar inner(const SymR2 & other) const;

  /// Norm squared
  Scalar norm_sq() const;

  /// Norm
  Scalar norm() const;

  /// Outer product ij,kl -> ijkl
  SymSymR4 outer(const SymR2 & other) const;

private:
  static constexpr TorchSize reverse_index[3][3] = {{0, 5, 4}, {5, 1, 3}, {4, 3, 2}};
};
