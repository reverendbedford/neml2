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

#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledVector.h"

using namespace torch::indexing;

namespace neml2
{
LabeledMatrix::LabeledMatrix(const LabeledTensor<1, 2> & other)
  : LabeledTensor<1, 2>(other)
{
}

LabeledMatrix
LabeledMatrix::zeros(TorchShapeRef batch_size,
                     const std::vector<const LabeledAxis *> & axes,
                     const torch::TensorOptions & options)
{
  return LabeledTensor<1, 2>::zeros(batch_size, axes, options);
}

LabeledMatrix
LabeledMatrix::identity(TorchSize nbatch,
                        const LabeledAxis & axis,
                        const torch::TensorOptions & options)
{
  return LabeledTensor<1, 2>(
      BatchTensor<1>::identity(axis.storage_size(), options).batch_expand_copy({nbatch}),
      {&axis, &axis});
}

void
LabeledMatrix::accumulate(const LabeledMatrix & other, bool recursive)
{
  for (auto [idxi, idxi_other] : LabeledAxis::common_indices(axis(0), other.axis(0), recursive))
    for (auto [idxj, idxj_other] : LabeledAxis::common_indices(axis(1), other.axis(1), recursive))
      _tensor.base_index({idxi, idxj}) += other.tensor().base_index({idxi_other, idxj_other});
}

void
LabeledMatrix::fill(const LabeledMatrix & other, bool recursive)
{
  for (auto [idxi, idxi_other] : LabeledAxis::common_indices(axis(0), other.axis(0), recursive))
    for (auto [idxj, idxj_other] : LabeledAxis::common_indices(axis(1), other.axis(1), recursive))
      _tensor.base_index({idxi, idxj}).copy_(other.tensor().base_index({idxi_other, idxj_other}));
}

LabeledMatrix
LabeledMatrix::chain(const LabeledMatrix & other) const
{
  // This function expresses a chain rule, which is just a dot product between the values of this
  // and the values of the input The main annoyance is just getting the names correct

  // Check that we are conformal
  neml_assert_dbg(batch_size() == other.batch_size(), "LabeledMatrix batch sizes are not the same");
  neml_assert_dbg(axis(1) == other.axis(0), "Labels are not conformal");

  // If all the sizes are correct then executing the chain rule is pretty easy
  return LabeledMatrix(torch::bmm(tensor(), other.tensor()), {&axis(0), &other.axis(1)});
}

LabeledMatrix
LabeledMatrix::inverse() const
{
  neml_assert_dbg(axis(0).storage_size() == axis(1).storage_size(),
                  "Can only invert square derivatives");

  return LabeledMatrix(torch::linalg::inv(tensor()), {&axis(1), &axis(0)});
}
} // namespace neml2
