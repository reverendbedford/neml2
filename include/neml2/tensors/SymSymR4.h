#pragma once

#include "neml2/tensors/FixedDimTensor.h"
#include "neml2/tensors/SymR2.h"

namespace neml2
{
/// A rank 4 tensor with minor symmetry stored in Mandel notation as a (6,6)
class SymSymR4 : public FixedDimTensor<1, 6, 6>
{
public:
  using FixedDimTensor<1, 6, 6>::FixedDimTensor;

  /// Named constructors
  /// @{
  enum FillMethod
  {
    identity_sym, // (dik,djl + dil,djk) / 2
    identity_vol, // dij,dkl / 3
    identity_dev, // dik,djl - dij,dkl / 3
    isotropic_E_nu
  };

  static SymSymR4 init(FillMethod method, const std::vector<Scalar> & vals = {});
  /// @}

  // Negation
  SymSymR4 operator-() const;

  // Inversion
  SymSymR4 inverse() const;

private:
  /// Helpers for the fill method
  /// @{
  static SymSymR4 init_identity();
  static SymSymR4 init_identity_sym();
  static SymSymR4 init_isotropic_E_nu(const Scalar & E, const Scalar & nu);
  /// @}
};

// We would like to have exact match for the basic operators to avoid ambiguity, and also to keep
// the return type as one of our supported primitive tensor types. All we need to do is to forward
// the calls to torch :)
/// @{
SymSymR4 operator+(const SymSymR4 & a, const SymSymR4 & b);

SymSymR4 operator-(const SymSymR4 & a, const SymSymR4 & b);

SymSymR4 operator*(const SymSymR4 & a, const Scalar & b);
SymSymR4 operator*(const Scalar & a, const SymSymR4 & b);
SymR2 operator*(const SymSymR4 & a, const SymR2 & b);
SymR2 operator*(const SymR2 & a, const SymSymR4 & b);
SymSymR4 operator*(const SymSymR4 & a, const SymSymR4 & b);

SymSymR4 operator/(const SymSymR4 & a, const Scalar & b);
/// @}
} // namespace neml2
