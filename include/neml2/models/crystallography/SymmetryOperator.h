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

#include "neml2/tensors/R2Base.h"

namespace neml2
{
class Rot;
class Quaternion;

namespace crystallography
{
/// @brief A generic symmetry operator
/// This includes the identity, proper rotations, improper rotations,
/// and inversion centers.  We also include reflections, even though
/// they are not proper operators themselves as they are useful in
/// defining others.
class SymmetryOperator : public R2Base<SymmetryOperator>
{
public:
  using R2Base<SymmetryOperator>::R2Base;

  /// Construct from quaternions, useful for comparison to old NEML
  static SymmetryOperator from_quaternion(const Quaternion & q);

  /// The identity transformation, i.e.e the Rank2 identity tensor
  static SymmetryOperator Identity(const torch::TensorOptions & options = default_tensor_options);
  /// A proper rotation, here provided by a Rot object
  static SymmetryOperator ProperRotation(const Rot & rot);
  /// An improper rotation (rotation + reflection), here provided by a rot object giving the rotation and reflection axis
  static SymmetryOperator ImproperRotation(const Rot & rot);
  /// A reflection, defined by the reflection plane
  static SymmetryOperator Reflection(const Vec & v);
  /// An inversion center
  static SymmetryOperator Inversion(const torch::TensorOptions & options = default_tensor_options);
};

/// @brief Composition as the multiplication operator
SymmetryOperator operator*(const SymmetryOperator & A, const SymmetryOperator & B);

} // namespace crystallography
} // namespace neml2