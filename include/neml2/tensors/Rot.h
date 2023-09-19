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

#include "neml2/tensors/Vec.h"

namespace neml2
{
// Forward declarations
class R2;
class SymR2;
class R3;
class R4;
class SymSymR4;

class RotRot;
class VecRot;
class R2Rot;
class R4Rot;
class SymR2Rot;
class SymSymR4Rot;

class Rot : public Vec
{
public:
  using Vec::Vec;

  /// Named constructors
  /// @{
  /// Construct from three scalar components
  static Rot init(const Scalar & r0, const Scalar & r1, const Scalar & r2);
  static Rot init(const Real & r0,
                  const Real & r1,
                  const Real & r2,
                  const torch::TensorOptions & options = default_tensor_options);
  /// The identity rotation, helpfully the zero vector
  static Rot identity(const torch::TensorOptions & options = default_tensor_options);
  /// @}

  /// The derivative of a rotation with respect to itself
  [[nodiscard]] static RotRot
  identity_map(const torch::TensorOptions & options = default_tensor_options);

  /// Inversion
  Rot inverse() const;

  /// Dot product
  using Vec::dot;

  /// Cross product
  using Vec::cross;

  /// Outer product
  using Vec::outer;

  /// Norm squared
  Scalar n2() const;

  /// Generate a rotation matrix using the Euler-Rodrigues formula
  R2 to_R2() const;

  /// d(to_R2)/d(r) -- useful in constructing other derivatives
  R3 dR2() const;

  /// Apply a rotation to various things
  /// @{
  /// Compose rotations in matrix convention: first R then this
  Rot apply(const Rot & R) const;
  /// Rotate a vector
  Vec apply(const Vec & v) const;
  /// Rotate a R2
  R2 apply(const R2 & T) const;
  /// Rotate a SymR2
  SymR2 apply(const SymR2 & T) const;
  /// Rotate a R4
  R4 apply(const R4 & T) const;
  /// Rotate a SymSymR4
  SymSymR4 apply(const SymSymR4 & T) const;
  /// @}

  /// Derivatives of the apply functions
  /// @{
  /// Composition of rotations
  RotRot dapply(const Rot & R) const;
  /// Vec rotation
  VecRot dapply(const Vec & v) const;
  /// R2 rotation
  R2Rot dapply(const R2 & T) const;
  /// SymR2 rotation
  SymR2Rot dapply(const SymR2 & T) const;
  /// R4 rotation
  R4Rot dapply(const R4 & T) const;
  /// SymSymR4 rotation
  SymSymR4Rot dapply(const SymSymR4 & T) const;
  /// @}
};

/// Composition of rotations r3 = r1 * r2 (r2 first, then r1)
//  So this follows the "matrix" convention where it's exactly the same
//  as the standard matrix product R1 * R2 where R1 and R2 are the
//  matrix representations of r1 and r2
Rot operator*(const Rot & r1, const Rot & r2);

} // namespace neml2
