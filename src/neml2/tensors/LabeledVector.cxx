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

#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/LabeledMatrix.h"

namespace neml2
{
LabeledVector::LabeledVector(const LabeledTensor<1> & other)
  : LabeledTensor<1>(other)
{
}

LabeledVector
LabeledVector::zeros(TorchShapeRef batch_size,
                     const std::vector<const LabeledAxis *> & axes,
                     const torch::TensorOptions & options)
{
  return LabeledTensor<1>::zeros(batch_size, axes, options);
}

LabeledVector
LabeledVector::zeros_like(const LabeledVector & other)
{
  return LabeledTensor<1>::zeros_like(other);
}

LabeledVector
LabeledVector::slice(const std::string & name) const
{
  return LabeledVector(_tensor.base_index({_axes[0]->indices(name)}), {&_axes[0]->subaxis(name)});
}

void
LabeledVector::accumulate(const LabeledVector & other, bool recursive)
{
  const auto indices = LabeledAxis::common_indices(axis(0), other.axis(0), recursive);
  for (const auto & [idx, idx_other] : indices)
    _tensor.base_index({idx}) += other.base_index({idx_other});
}

void
LabeledVector::fill(const LabeledVector & other, bool recursive)
{
  const auto indices = LabeledAxis::common_indices(axis(0), other.axis(0), recursive);
  for (const auto & [idx, idx_other] : indices)
    _tensor.base_index({idx}).copy_(other.base_index({idx_other}));
}

LabeledMatrix
LabeledVector::outer(const LabeledVector & other) const
{
  return LabeledMatrix(math::bmm(tensor().base_unsqueeze(-1), other.tensor().base_unsqueeze(-2)),
                       {&axis(0), &other.axis(0)});
}

namespace utils
{
bool
allclose(const LabeledVector & a, const LabeledVector & b, Real rtol, Real atol)
{
  if (a.axis(0) != b.axis(0))
    return false;

  for (auto var : a.axis(0).variable_accessors(true))
    if (!torch::allclose(a(var), b(var), rtol, atol))
      return false;

  return true;
}
}
} // namespace neml2
