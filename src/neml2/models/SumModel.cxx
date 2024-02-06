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

#include "neml2/models/SumModel.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(ScalarSumModel);
register_NEML2_object(SR2SumModel);

template <typename T>
OptionSet
SumModel<T>::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<std::vector<LabeledAxisAccessor>>("from_var");
  options.set<LabeledAxisAccessor>("to_var");
  return options;
}

template <typename T>
SumModel<T>::SumModel(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<T>(options.get<LabeledAxisAccessor>("to_var")))
{
  for (auto fv : options.get<std::vector<LabeledAxisAccessor>>("from_var"))
    _from.push_back(&declare_input_variable<T>(fv));
}

template <typename T>
void
SumModel<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
  {
    auto sum = T::zeros_like(_to);
    for (auto fv : _from)
      sum += *fv;
    _to = sum;
  }

  if (dout_din)
    for (auto fv : _from)
      _to.d(*fv) = T::identity_map(options());

  if (d2out_din2)
  {
    // zero
  }
}

template class SumModel<Scalar>;
template class SumModel<SR2>;
} // namespace neml2
