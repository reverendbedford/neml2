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
class Rotation : public FixedDimTensor<1, 3>
{
public:
  using FixedDimTensor<1, 3>::FixedDimTensor;

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
};

/// Composition of rotations r3 = r1 * r2 (r2 first, then r1)
Rotation operator*(const Rotation & r1, const Rotation & r2);

} // namespace neml2
