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

namespace neml2
{
register_NEML2_object(Normality);

OptionSet
Normality::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<std::string>("model");
  options.set<VariableName>("function");
  options.set<std::vector<VariableName>>("from");
  options.set<std::vector<VariableName>>("to");
  return options;
}

Normality::Normality(const OptionSet & options)
  : Model(options),
    _model(register_model<Model>(options.get<std::string>("model"), /*extra_deriv_order=*/1)),
    _f(options.get<VariableName>("function"))
{
  // Set up the conjugate pairs
  const auto from = options.get<std::vector<VariableName>>("from");
  const auto to = options.get<std::vector<VariableName>>("to");
  neml_assert(from.size() == to.size(),
              "The conjugate pairs should have a one-to-one correspondance. ",
              from.size(),
              " variables are being mapped to ",
              to.size(),
              " variables.");
  for (size_t i = 0; i < from.size(); i++)
  {
    auto sz = _model.output_axis().storage_size(_f) * _model.input_axis().storage_size(from[i]);
    _conjugate_pairs.emplace(from[i], &declare_output_variable(sz, to[i]));
  }
}

void
Normality::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Normality doesn't implement second derivatives.");

  // All we do here is simply mapping the derivatives.
  // However, let's consider all the cases to make it as efficient as possible.
  if (out && !dout_din)
    _model.value_and_dvalue();
  else
    _model.value_and_dvalue_and_d2value();

  for (auto && [ivar, var] : _conjugate_pairs)
  {
    if (out)
      (*var) = _model.derivative_storage()(_f, ivar);

    if (dout_din)
      for (auto && [jvar, j] : input_views())
        var->d(j) = _model.second_derivative_storage()(_f, ivar, jvar);
  }
}
} // namespace neml2
