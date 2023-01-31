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

#include "neml2/models/SecDerivModel.h"

namespace neml2
{
ParameterSet
SecDerivModel::expected_params()
{
  ParameterSet params = Model::expected_params();
  return params;
}

LabeledMatrix
SecDerivModel::dvalue(LabeledVector in) const
{
  LabeledMatrix dout_din(in.batch_size(), output(), in.axis(0));
  set_dvalue(in, dout_din);
  return dout_din;
}

LabeledTensor<1, 3>
SecDerivModel::d2value(LabeledVector in) const
{
  auto [dout_din, d2out_din2] = dvalue_and_d2value(in);
  return d2out_din2;
}

std::tuple<LabeledMatrix, LabeledTensor<1, 3>>
SecDerivModel::dvalue_and_d2value(LabeledVector in) const
{
  LabeledMatrix dout_din(in.batch_size(), output(), in.axis(0));
  LabeledTensor<1, 3> d2out_din2(in.batch_size(), output(), in.axis(0), in.axis(0));
  set_dvalue(in, dout_din, &d2out_din2);
  return {dout_din, d2out_din2};
}
} // namespace neml2
