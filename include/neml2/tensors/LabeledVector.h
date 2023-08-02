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

#include "neml2/tensors/LabeledTensor.h"

namespace neml2
{
class LabeledMatrix;

class LabeledVector : public LabeledTensor<1, 1>
{
public:
  using LabeledTensor<1, 1>::LabeledTensor;

  /// Conversion from a LabeledTensor
  LabeledVector(const LabeledTensor<1, 1> & other);

  /// Since we assume a flat batch for now, we can define this convenient method to retrive the single batch size.
  TorchSize batch_size() const { return tensor().batch_sizes()[0]; }

  /// Slice the logically 1D tensor by a single sub-axis
  LabeledVector slice(const std::string & name) const;

  /// Add another vector into this vector.
  /// The item set of the other vector must be a subset of this vector's item set.
  void accumulate(const LabeledVector & other, bool recursive = true);

  /// Fill (override) another vector into this vector.
  /// The item set of the other vector must be a subset of this vector's item set.
  void fill(const LabeledVector & other, bool recursive = true);

  /// Promote to LabeledMatrix with arbitrary outer product
  LabeledMatrix outer(const LabeledVector & other) const;
};

namespace utils
{
bool allclose(const LabeledVector & a, const LabeledVector & b, Real rtol = 1e-5, Real atol = 1e-8);
} // namespaca utils
} // namespace neml2
