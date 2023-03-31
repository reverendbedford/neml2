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
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
register_NEML2_object(ScalarSumModel);
register_NEML2_object(SymR2SumModel);

template <typename T>
ParameterSet
SumModel<T>::expected_params()
{
  ParameterSet params = Model::expected_params();
  params.set<std::vector<LabeledAxisAccessor>>("from_var");
  params.set<LabeledAxisAccessor>("to_var");
  return params;
}

template <typename T>
SumModel<T>::SumModel(const ParameterSet & params)
  : Model(params),
    to(declare_output_variable<T>(params.get<LabeledAxisAccessor>("to_var")))
{
  for (auto fv : params.get<std::vector<LabeledAxisAccessor>>("from_var"))
    from.push_back(declare_input_variable<T>(fv));

  this->setup();
}

template <typename T>
void
SumModel<T>::set_value(LabeledVector in,
                       LabeledVector * out,
                       LabeledMatrix * dout_din,
                       LabeledTensor3D * d2out_din2) const
{
  const auto options = in.options();
  const auto nbatch = in.batch_size();

  if (out)
  {
    auto sum = T::zero(options).batch_expand_copy(nbatch);
    for (auto fv : from)
      sum += in(fv);
    out->set(sum, to);
  }

  if (dout_din)
  {
    for (auto fv : from)
      dout_din->set(T::identity_map(options).batch_expand(nbatch), to, fv);
  }

  if (d2out_din2)
  {
    // zero
  }
}

template class SumModel<Scalar>;
template class SumModel<SymR2>;
} // namespace neml2
