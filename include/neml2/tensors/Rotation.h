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

#include "neml2/tensors/Vector.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SymR2.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
class Rotation : public Vector
{
public:
  using Vector::Vector;

  /// Named constructors
  /// @{
  /// Construct from three scalar components
  static Rotation init(const Scalar & r0, const Scalar & r1, const Scalar & r2);
  /// The identity rotation, helpfully the zero vector
  static Rotation identity();
  /// @}

  /// Accessor
  Scalar operator()(TorchSize i) const;

  /// Inversion
  Rotation inverse() const;

  /// Dot product
  using Vector::dot;

  /// Cross product
  using Vector::cross;

  /// Generate a rotation matrix
  // Using the Euler-Rodrigues formula
  R2 to_R2() const;

  /// Apply a rotation to various things
  /// @{
  /// Compose rotations in matrix convention: first R then this
  Rotation apply(const Rotation & R) const;
  /// Rotate a vector
  Vector apply(const Vector & v) const;
  /// Rotate a R2
  R2 apply(const R2 & T) const;
  /// Rotate a SymR2
  SymR2 apply(const SymR2 & T) const;
  /// Rotate a R4
  R4 apply(const R4 & T) const;
  /// Rotate a SymSymR4
  SymSymR4 apply(const SymSymR4 & T) const;
  /// @}
};

/// Composition of rotations r3 = r1 * r2 (r2 first, then r1)
//  So this follows the "matrix" convention where it's exactly the same
//  as the standard matrix product R1 * R2 where R1 and R2 are the
//  matrix representations of r1 and r2
Rotation operator*(const Rotation & r1, const Rotation & r2);

} // namespace neml2