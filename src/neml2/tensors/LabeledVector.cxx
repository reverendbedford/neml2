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
LabeledVector::LabeledVector(const LabeledTensor<1, 1> & other)
  : LabeledTensor<1, 1>(other)
{
}

LabeledVector
LabeledVector::slice(const std::string & name) const
{
  return LabeledVector(_tensor.base_index({_axes[0]->indices(name)}), _axes[0]->subaxis(name));
}

void
LabeledVector::accumulate(const LabeledVector & other, bool recursive)
{
  auto [idx, idx_other] = LabeledAxis::common_indices(axis(0), other.axis(0), recursive);
  _tensor.base_index_put({idx}, _tensor.base_index({idx}) + other.tensor().base_index({idx_other}));
}

void
LabeledVector::fill(const LabeledVector & other, bool recursive)
{
  auto [idx, idx_other] = LabeledAxis::common_indices(axis(0), other.axis(0), recursive);
  _tensor.base_index_put({idx}, other.tensor().base_index({idx_other}));
}

LabeledMatrix
LabeledVector::outer(const LabeledVector & other) const
{
  return LabeledMatrix(
      torch::bmm(tensor().unsqueeze(2), other.tensor().unsqueeze(1)), axis(0), other.axis(0));
}

void
LabeledVector::write(std::ostream & os, std::string delimiter, TorchSize batch, bool header) const
{
  if (header)
  {
    for (auto name : axis(0).item_names())
    {
      TorchSize sz = axis(0).storage_size(name);
      if (sz == 0)
        continue;
      else if (sz == 1)
        os << delimiter << name;
      else
        for (TorchSize i = 0; i < sz; i++)
          os << delimiter << name << "_" << i;
    }
    os << std::endl;
  }

  for (auto name : axis(0).item_names())
  {
    TorchSize sz = axis(0).storage_size(name);
    if (sz == 0)
      continue;
    else
      for (TorchSize i = 0; i < sz; i++)
        os << delimiter << (*this)(name).index({batch, i}).item<double>();
  }
  os << std::endl;
}
} // namespace neml2
