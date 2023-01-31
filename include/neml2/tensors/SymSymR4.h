// Copyright 2023, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "neml2/tensors/FixedDimTensor.h"
#include "neml2/tensors/SymR2.h"

namespace neml2
{
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

SymSymR4 operator+(const SymSymR4 & a, const SymSymR4 & b);

SymSymR4 operator-(const SymSymR4 & a, const SymSymR4 & b);

SymSymR4 operator*(const SymSymR4 & a, const Scalar & b);
SymSymR4 operator*(const Scalar & a, const SymSymR4 & b);
SymR2 operator*(const SymSymR4 & a, const SymR2 & b);
SymR2 operator*(const SymR2 & a, const SymSymR4 & b);
SymSymR4 operator*(const SymSymR4 & a, const SymSymR4 & b);

SymSymR4 operator/(const SymSymR4 & a, const Scalar & b);
} // namespace neml2
