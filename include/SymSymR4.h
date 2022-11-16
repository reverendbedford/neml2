#pragma once

#include "FixedDimTensor.h"

#include "SymR2.h"

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

  /// Tensor product ijkl,kl->ij
  SymR2 operator*(const SymR2 & b);

  /// Tensor product ijkl,klmn->ijmn
  SymSymR4 operator*(const SymSymR4 & b);

private:
  /// Helpers for the fill method
  /// @{
  static SymSymR4 init_identity();
  static SymSymR4 init_identity_sym();
  static SymSymR4 init_isotropic_E_nu(const Scalar & E, const Scalar & nu);
  /// @}
};
