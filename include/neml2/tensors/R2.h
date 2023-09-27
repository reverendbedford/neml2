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

namespace neml2
{
class Scalar;
class Vec;
class SR2;
class R3;
class Rot;

class R2 : public FixedDimTensor<R2, 3, 3>
{
public:
  using FixedDimTensor<R2, 3, 3>::FixedDimTensor;

  R2(const SR2 & S);

  explicit R2(const Rot & r);

  /// Conversion operator to symmetric second order tensor
  explicit operator SR2() const;

  /// Fill the diagonals with a11 = a22 = a33 = a
  [[nodiscard]] static R2 fill(const Real & a,
                               const torch::TensorOptions & options = default_tensor_options);
  [[nodiscard]] static R2 fill(const Scalar & a);
  /// Fill the diagonals with a11, a22, a33
  [[nodiscard]] static R2 fill(const Real & a11,
                               const Real & a22,
                               const Real & a33,
                               const torch::TensorOptions & options = default_tensor_options);
  [[nodiscard]] static R2 fill(const Scalar & a11, const Scalar & a22, const Scalar & a33);
  /// Fill symmetric entries
  [[nodiscard]] static R2 fill(const Real & a11,
                               const Real & a22,
                               const Real & a33,
                               const Real & a23,
                               const Real & a13,
                               const Real & a12,
                               const torch::TensorOptions & options = default_tensor_options);
  [[nodiscard]] static R2 fill(const Scalar & a11,
                               const Scalar & a22,
                               const Scalar & a33,
                               const Scalar & a23,
                               const Scalar & a13,
                               const Scalar & a12);
  /// Fill all entries
  [[nodiscard]] static R2 fill(const Real & a11,
                               const Real & a12,
                               const Real & a13,
                               const Real & a21,
                               const Real & a22,
                               const Real & a23,
                               const Real & a31,
                               const Real & a32,
                               const Real & a33,
                               const torch::TensorOptions & options = default_tensor_options);
  [[nodiscard]] static R2 fill(const Scalar & a11,
                               const Scalar & a12,
                               const Scalar & a13,
                               const Scalar & a21,
                               const Scalar & a22,
                               const Scalar & a23,
                               const Scalar & a31,
                               const Scalar & a32,
                               const Scalar & a33);
  /// Skew matrix from Vec
  [[nodiscard]] static R2 skew(const Vec & v);
  /// Identity
  [[nodiscard]] static R2 identity(const torch::TensorOptions & options = default_tensor_options);

  /// Rotate
  R2 rotate(const Rot & r) const;

  /// Derivative of the rotated tensor w.r.t. the Rodrigues vector
  R3 drotate(const Rot & r) const;

  /// Accessor
  Scalar operator()(TorchSize i, TorchSize j) const;

  /// Inversion
  R2 inverse() const;

  /// transpose
  R2 transpose() const;
};

R2 operator*(const R2 & A, const R2 & B);
Vec operator*(const R2 & A, const Vec & b);
} // namespace neml2
