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
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vector.h"
#include "neml2/tensors/SymR2.h"

namespace neml2
{
class Vector;
class SymR2;

class R2 : public FixedDimTensor<1, 3, 3>
{
public:
  using FixedDimTensor<1, 3, 3>::FixedDimTensor;

  /// Named constructors
  /// @{
  /// From a SymR2
  static R2 init(const SymR2 & sym);
  /// Skew matrix from vector
  static R2 init(const Vector & v);
  /// Identity
  static R2 identity();
  /// All zeros
  static R2 zero();
  /// @}

  /// Accessor
  Scalar operator()(TorchSize i, TorchSize j) const;

  /// transpose
  R2 transpose() const;

  /// symmetrize
  SymR2 to_symmetric() const;
};

/// Matrix-matrix product
R2 operator*(const R2 & A, const R2 & B);

/// Matrix-vector product
Vector operator*(const R2 & A, const Vector & b);

/// Matrix-scalar product
R2 operator*(const R2 & A, const Scalar & b);

/// Matrix-scalar product
R2 operator*(const Scalar & a, const R2 & B);

} // namespace neml2
