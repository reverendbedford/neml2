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

#include "neml2/tensors/LabeledTensor3D.h"
#include "neml2/tensors/LabeledMatrix.h"

using namespace torch::indexing;

namespace neml2
{
LabeledTensor3D::LabeledTensor3D(const LabeledTensor<1, 3> & other)
  : LabeledTensor<1, 3>(other)
{
}

LabeledTensor3D
LabeledTensor3D::zeros(TorchShapeRef batch_size,
                       const std::vector<const LabeledAxis *> & axes,
                       const torch::TensorOptions & options)
{
  return LabeledTensor<1, 3>::zeros(batch_size, axes, options);
}

LabeledTensor3D
LabeledTensor3D::zeros_like(const LabeledTensor3D & other)
{
  return LabeledTensor<1, 3>::zeros_like(other);
}

void
LabeledTensor3D::accumulate(const LabeledTensor3D & other, bool recursive)
{
  for (auto [idxi, idxi_other] : LabeledAxis::common_indices(axis(0), other.axis(0), recursive))
    for (auto [idxj, idxj_other] : LabeledAxis::common_indices(axis(1), other.axis(1), recursive))
      for (auto [idxk, idxk_other] : LabeledAxis::common_indices(axis(2), other.axis(2), recursive))
        _tensor.base_index({idxi, idxj, idxk}) +=
            other.tensor().base_index({idxi_other, idxj_other, idxk_other});
}

void
LabeledTensor3D::fill(const LabeledTensor3D & other, bool recursive)
{
  for (auto [idxi, idxi_other] : LabeledAxis::common_indices(axis(0), other.axis(0), recursive))
    for (auto [idxj, idxj_other] : LabeledAxis::common_indices(axis(1), other.axis(1), recursive))
      for (auto [idxk, idxk_other] : LabeledAxis::common_indices(axis(2), other.axis(2), recursive))
        _tensor.base_index({idxi, idxj, idxk})
            .copy_(other.tensor().base_index({idxi_other, idxj_other, idxk_other}));
}

LabeledTensor3D
LabeledTensor3D::chain(const LabeledTensor3D & other,
                       const LabeledMatrix & dself,
                       const LabeledMatrix & dother) const
{
  // This function expresses the second oreder chain rule, which can be expressed as
  // d2y/dx2 = d2y/du2 du/dx du/dx + dy/du d2u/dx2
  // In index notation this is
  // (d2y/dx2)_{ijk} = (d2y/du2)_{ipq} (du/dx)_{pj} (du/dx)_{qk} + (dy/du)_{ip} (d2u/dx2)_{pjk}

  // Make sure we are conformal
  neml_assert_dbg(batch_size() == other.batch_size(), "Batch sizes are not the same");
  neml_assert_dbg(batch_size() == dself.batch_size(), "Batch sizes are not the same");
  neml_assert_dbg(batch_size() == dother.batch_size(), "Batch sizes are not the same");
  neml_assert_dbg(axis(1) == axis(2), "Self labels are not conformal");
  neml_assert_dbg(other.axis(1) == other.axis(2), "Other labels are not conformal");
  neml_assert_dbg(axis(2) == other.axis(0), "Self and other labels are not conformal");

  // If all the sizes are correct then executing the chain rule is pretty easy
  return LabeledTensor3D(einsum({tensor(), dother.tensor(), dother.tensor()}, {"ipq", "pj", "qk"}) +
                             einsum({dself.tensor(), other.tensor()}, {"ip", "pjk"}),
                         {&axis(0), &other.axis(1), &other.axis(2)});
}
} // namespace neml2
