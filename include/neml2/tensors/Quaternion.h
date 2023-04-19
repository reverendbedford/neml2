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

namespace neml2
{
class Quaternion : public FixedDimTensor<1, 4>
{
public:
  using FixedDimTensor<1, 4>::FixedDimTensor;

  /// Named constructors
  /// @{
  /// Construct from four scalar components
  static Quaternion
  init(const Scalar & q0, const Scalar & q1, const Scalar & q2, const Scalar & q3);
  /// Identity
  static Quaternion identity();
  /// @}

  /// Accessor
  Scalar operator()(TorchSize i) const;

  /// Negation
  Quaternion operator-() const;

  /// Conjugation
  Quaternion conj() const;

  /// Quaternion dot product, useful for distances
  Scalar inner(const Quaternion & other) const;

  /// Squared quaternion norm
  Scalar norm_sq() const;

  /// Quaternion norm
  Scalar norm() const;

  /// Square norm of the vector part
  Scalar normv_sq() const;

  /// Norm of the vector part
  Scalar normv() const;

  /// Inversion
  Quaternion inverse() const;
};

/// Scalar multiplication
/// @{
Quaternion operator*(const Scalar & a, const Quaternion & b);
Quaternion operator*(const Quaternion & a, const Scalar & b);
/// @}

/// Scalar division
/// @{
Quaternion operator/(const Quaternion & a, const Scalar & b);
/// @}

/// Quaternion composition
Quaternion operator*(const Quaternion & a, const Quaternion & b);

/// Quaternion exponential map
Quaternion exp(const Quaternion & a);

/// Quaternion inverse exponential map
Quaternion log(const Quaternion & a);

} // namespace neml2
