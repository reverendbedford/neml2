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

  /// accessor for the first lattice vector
  Vec a1() const;
  /// accessor for the second lattice vector
  Vec a2() const;
  /// accessor for the third lattice vector
  Vec a3() const;

  /// accessor for the first reciprocal lattice vector
  Vec b1() const;
  /// accessor for the second reciprocal lattice vector
  Vec b2() const;
  /// accessor for the third reciprocal lattice vector
  Vec b3() const;

  /// Slice a BatchTensor to give a group of slip data
  // The slice happens along the last batch axis
  template <typename T>
  T slip_slice(const T & tensor, TorchSize grp) const;

private:
  /// Delegated constructor to setup schmid tensors and slice indices at once
  CrystalGeometry(const OptionSet & options, std::pair<R2, std::vector<TorchSize>> slip_res);

  /// Helper to setup reciprocal lattice
  static Vec make_reciprocal_lattice(const Vec & lattice_vectors);

  /// Helper to setup slip systems
  static std::pair<R2, std::vector<TorchSize>>
  setup_schmid_tensors(const CrystalClass & cls,
                       const MillerIndex & slip_directions,
                       const MillerIndex & slip_planes);

private:
  /// Crystal symmetry class with operators
  const CrystalClass & _class;
  /// Lattice vectors
  const Vec & _lattice_vectors;
  /// Reciprocal lattice vectors
  const Vec & _reciprocal_lattice_vectors;
  /// Slip directions
  const MillerIndex & _slip_directions;
  /// Slip planes
  const MillerIndex & _slip_planes;

  /// Output: full Schmid tensor for each slip system
  const R2 & _A;
  /// Offsets into batch tensors for each slip group
  const std::vector<TorchSize> _slip_offsets;
  /// Output: symmetric Schmid tensors for each slip system
  const SR2 & _M;
  /// Output: skew-symmetry Schmid tensors for each slip system
  const WR2 & _W;
};

template <typename T>
T
CrystalGeometry::slip_slice(const T & tensor, TorchSize grp) const
{
  (void)grp;
  return tensor;
}

/// Helper that reduces out equivalent cartesian directions, this version considers equivalence
/// to be in either direction
Vec unique_bidirectional(const Vec & inp);

} // namespace crystallography
} // namespace neml2