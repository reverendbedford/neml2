// Copyright 2024, UChicago Argonne, LLC
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
/**
 * @brief A single-batched, logically 2D LabeledTensor.
 *
 */
class LabeledMatrix : public LabeledTensor<LabeledMatrix, 2>
{
public:
  using LabeledTensor<LabeledMatrix, 2>::LabeledTensor;

  /// Create a labeled identity tensor
  static LabeledMatrix identity(TensorShapeRef batch_size,
                                const LabeledAxis & axis,
                                const torch::TensorOptions & options = default_tensor_options());

  /// Fill another matrix into this matrix.
  void fill(const LabeledMatrix & other);

  /// Split the matrix by subaxes along a given dimension
  std::map<LabeledAxisAccessor, LabeledMatrix> split(Size dim) const;

  /// Disassemble the matrix into a matrix of matrices (by variables)
  std::map<LabeledAxisAccessor, std::map<LabeledAxisAccessor, Tensor>> disassemble() const;

  /// Assemble a matrix of matrices
  static LabeledMatrix
  assemble(const std::map<LabeledAxisAccessor, std::map<LabeledAxisAccessor, Tensor>> & vals_dict,
           const LabeledAxis & yaxis,
           const LabeledAxis & xaxis);
};
} // namespace neml2
