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

/**
 * @brief A single-batched, logically 3D LabeledTensor.
 *
 */
class LabeledTensor3D : public LabeledTensor<1, 3>
{
public:
  using LabeledTensor<1, 3>::LabeledTensor;

  /// Conversion from LabeledTensor
  LabeledTensor3D(const LabeledTensor<1, 3> & other);

  /// Create a batched, labeled zero tensor
  static LabeledTensor3D zeros(TorchShapeRef batch_size,
                               const std::vector<const LabeledAxis *> & axes,
                               const torch::TensorOptions & options = default_tensor_options);

  /// Create a zero tensor with the same shape and labels as the other tensor
  static LabeledTensor3D zeros_like(const LabeledTensor3D & other);

  /// Since we assume a flat batch for now, we can define this convenient method to retrive the single batch size.
  TorchSize batch_size() const { return tensor().batch_sizes()[0]; }

  /// Add another tensor into this tensor.
  /// The item set of the other tensor must be a subset of this tensor's item set.
  void accumulate(const LabeledTensor3D & other, bool recursive = true);

  /// Fill another tensor into this tensor.
  /// The item set of the other tensor must be a subset of this tensor's item set.
  void fill(const LabeledTensor3D & other, bool recursive = true);

  /// Second order chain rule product of two derivatives
  LabeledTensor3D chain(const LabeledTensor3D & other,
                        const LabeledMatrix & dself,
                        const LabeledMatrix & dother) const;
};
} // namespace neml2
