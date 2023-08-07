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
class LabeledVector;

class LabeledMatrix : public LabeledTensor<1, 2>
{
public:
  using LabeledTensor<1, 2>::LabeledTensor;

  /// Conversion from LabeledTensor
  LabeledMatrix(const LabeledTensor<1, 2> & other);

  static LabeledMatrix zeros(TorchShapeRef batch_size,
                             const std::vector<const LabeledAxis *> & axes,
                             const torch::TensorOptions & options = default_tensor_options);

  /// Construct an identity
  static LabeledMatrix identity(TorchSize nbatch,
                                const LabeledAxis & axis,
                                const torch::TensorOptions & options = default_tensor_options);

  /// Since we assume a flat batch for now, we can define this convenient method to retrive the single batch size.
  TorchSize batch_size() const { return tensor().batch_sizes()[0]; }

  /// Add another matrix into this matrix.
  /// The item set of the other matrix must be a subset of this matrix's item set.
  void accumulate(const LabeledMatrix & other, bool recursive = true);

  /// Fill another matrix into this matrix.
  /// The item set of the other matrix must be a subset of this matrix's item set.
  void fill(const LabeledMatrix & other, bool recursive = true);

  /// Chain rule product of two derivatives
  LabeledMatrix chain(const LabeledMatrix & other) const;

  /// Invert a LabeledMatrix for use in an implicit function derivative
  LabeledMatrix inverse() const;
};
} // namespace neml2
