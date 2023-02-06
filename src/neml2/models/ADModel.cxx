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

#include "neml2/models/ADModel.h"

namespace neml2
{
ParameterSet
ADModel::expected_params()
{
  ParameterSet params = Model::expected_params();
  return params;
}

std::tuple<LabeledVector, LabeledMatrix>
ADModel::value_and_dvalue(LabeledVector in) const
{
  bool was_ad = true;
  if (!in.tensor().requires_grad())
  {
    was_ad = false;
    in.tensor().requires_grad_();
  }

  // Evalute the model (not its derivatives)
  auto out = Model::value(in);

  // Allocate space for Jacobian
  LabeledMatrix dout_din(out, in);

  // Loop over rows of the state to retrieve the derivatives
  for (TorchSize i = 0; i < out.tensor().base_sizes()[0]; i++)
  {
    BatchTensor<1> grad_outputs = torch::zeros_like(out.tensor());
    grad_outputs.index_put_({torch::indexing::Ellipsis, i}, 1);
    auto jac_row = torch::autograd::grad({out.tensor()}, {in.tensor()}, {grad_outputs}, true)[0];
    dout_din.tensor().base_index_put({i, torch::indexing::Slice()}, jac_row);
  }

  in.tensor().requires_grad_(was_ad);

  return {out, dout_din};
}
} // namespace neml2
