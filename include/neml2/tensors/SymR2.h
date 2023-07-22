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

#include "neml2/tensors/Scalar.h"
#include "neml2/misc/utils.h"

namespace neml2
{
class SymSymR4;

class SymR2 : public FixedDimTensor<1, 6>
{
public:
  using FixedDimTensor<1, 6>::FixedDimTensor;

  /// The derivative of a SymR2 with respect to itself
  [[nodiscard]] static SymSymR4
  identity_map(const torch::TensorOptions & options = default_tensor_options);

  /// Named constructors
  /// @{
  /// Make zero with batch size 1
  static SymR2 zero(const torch::TensorOptions & options = default_tensor_options);
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
  static SymR2 identity(const torch::TensorOptions & options = default_tensor_options);
  /// @}

  /// Accessor
  Scalar operator()(TorchSize i, TorchSize j) const;

  /// Negation
  SymR2 operator-() const;

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
  Scalar norm(Real eps = 0) const;

  /// Outer product ij,kl -> ijkl
  SymSymR4 outer(const SymR2 & other) const;

  /// Inversion
  SymR2 inverse() const;
};

SymR2 operator+(const SymR2 & a, const Scalar & b);
SymR2 operator+(const Scalar & a, const SymR2 & b);
SymR2 operator+(const SymR2 & a, const SymR2 & b);

SymR2 operator-(const SymR2 & a, const Scalar & b);
SymR2 operator-(const Scalar & a, const SymR2 & b);
SymR2 operator-(const SymR2 & a, const SymR2 & b);

SymR2 operator*(const SymR2 & a, const Scalar & b);
SymR2 operator*(const Scalar & a, const SymR2 & b);

SymR2 operator/(const SymR2 & a, const Scalar & b);
SymR2 operator/(const Scalar & a, const SymR2 & b);
} // namespace neml2
