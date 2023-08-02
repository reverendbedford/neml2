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

#include "neml2/models/solid_mechanics/Normality.h"
#include "neml2/models/solid_mechanics/YieldFunction.h"

namespace neml2
{
register_NEML2_object(Normality);

ParameterSet
Normality::expected_params()
{
  ParameterSet params = Model::expected_params();
  params.set<std::string>("model");
  params.set<LabeledAxisAccessor>("function");
  params.set<std::vector<LabeledAxisAccessor>>("from");
  params.set<std::vector<LabeledAxisAccessor>>("to");
  return params;
}

Normality::Normality(const ParameterSet & params)
  : Model(params),
    function(params.get<LabeledAxisAccessor>("function")),
    _model(Factory::get_object<Model>("Models", params.get<std::string>("model")))
{
  register_model(Factory::get_object_ptr<Model>("Models", params.get<std::string>("model")));

  // Set up the conjugate pairs
  const auto from = params.get<std::vector<LabeledAxisAccessor>>("from");
  const auto to = params.get<std::vector<LabeledAxisAccessor>>("to");
  neml_assert(from.size() == to.size(),
              "The conjugate pairs should have a one-to-one correspondance. ",
              from.size(),
              " variables are being mapped to ",
              to.size(),
              " variables.");
  for (size_t i = 0; i < from.size(); i++)
  {
    _conjugate_pairs.emplace(from[i], to[i]);
    auto sz = _model.output().storage_size(function) * _model.input().storage_size(from[i]);
    declare_output_variable(sz, to[i]);
  }

  setup();
}

void
Normality::set_value(const LabeledVector & in,
                     LabeledVector * out,
                     LabeledMatrix * dout_din,
                     LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "Normality doesn't implement second derivatives.");

  // All we do here is simply mapping the derivatives.
  // However, let's consider all the cases to make it as efficient as possible.
  if (out && !dout_din)
  {
    const auto df_din = _model.dvalue(in);
    for (const auto & [from, to] : _conjugate_pairs)
      out->set(df_din(function, from), to);
  }
  else if (!out && dout_din)
  {
    const auto d2f_din2 = _model.d2value(in);
    for (const auto & [from_i, to_i] : _conjugate_pairs)
      for (const auto & [from_j, to_j] : _conjugate_pairs)
        dout_din->set(d2f_din2(function, from_i, from_j), to_i, from_j);
  }
  else if (out && dout_din)
  {
    const auto [df_din, d2f_din2] = _model.dvalue_and_d2value(in);
    for (const auto & [from_i, to_i] : _conjugate_pairs)
    {
      out->set(df_din(function, from_i), to_i);
      for (const auto & [from_j, to_j] : _conjugate_pairs)
        dout_din->set(d2f_din2(function, from_i, from_j), to_i, from_j);
    }
  }
}
} // namespace neml2
