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

#include "neml2/models/StaticModel.h"

namespace neml2
{
class Vec;
class R2;
class SR2;
class WR2;
namespace crystallography
{
class CrystalClass;
class MillerIndex;

class CrystalGeometry : public StaticModel
{
public:
  /// Setup from parameter set
  CrystalGeometry(const OptionSet & options);

  /// Input options
  static OptionSet expected_options();

  /// Slice a BatchTensor to give a group of slip data
  // The slice happens along the last batch axis
  template <typename T>
  T slip_slice(const T & tensor, TorchSize grp);

private:
  /// Important helper to setup slip systems
  static R2 setup_schmid_tensors(const MillerIndex & slip_directions,
                                 const MillerIndex & slip_planes);

private:
  /// Crystal symmetry class with operators
  const CrystalClass & _class;
  /// Lattice vectors
  const Vec & _lattice_vectors;
  /// Slip directions
  const MillerIndex & _slip_directions;
  /// Slip planes
  const MillerIndex & _slip_planes;

  /// Output: full Schmid tensor for each slip system
  const R2 & _A;
  /// Output: symmetric Schmid tensors for each slip system
  const SR2 & _M;
  /// Output: skew-symmetry Schmid tensors for each slip system
  const WR2 & _W;
};

template <typename T>
T
CrystalGeometry::slip_slice(const T & tensor, TorchSize grp)
{
  (void)grp;
  return tensor;
}

} // namespace crystallography
} // namespace neml2