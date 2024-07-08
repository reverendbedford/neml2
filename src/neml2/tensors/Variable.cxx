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

#include "neml2/tensors/Variable.h"

namespace neml2
{
void
VariableBase::cache(TorchShapeRef batch_shape)
{
  _batch_sizes = batch_shape.vec();
}

StorageTensor<2>::View<BatchTensor> &
VariableBase::d(const VariableBase & x)
{
  neml_assert_dbg(_deriv_views.count(x.name()),
                  "Error retrieving first derivative: ",
                  name(),
                  " does not depend on ",
                  x.name());
  return _deriv_views[x.name()];
}

StorageTensor<3>::View<BatchTensor> &
VariableBase::d(const VariableBase & x1, const VariableBase & x2)
{
  neml_assert_dbg(_sec_deriv_views.count(x1.name()),
                  "Error retrieving second derivative: ",
                  name(),
                  " does not depend on ",
                  x1.name());
  neml_assert_dbg(_sec_deriv_views[x1.name()].count(x2.name()),
                  "Error retrieving second derivative: d(",
                  name(),
                  ")/d(",
                  x1.name(),
                  ") does not depend on ",
                  x2.name());
  return _sec_deriv_views[x1.name()][x2.name()];
}

const std::map<VariableName, StorageTensor<2>::View<BatchTensor>> &
VariableBase::derivatives()
{
  return _deriv_views;
}

const std::map<VariableName, std::map<VariableName, StorageTensor<3>::View<BatchTensor>>> &
VariableBase::second_derivatives()
{
  return _sec_deriv_views;
}
}
