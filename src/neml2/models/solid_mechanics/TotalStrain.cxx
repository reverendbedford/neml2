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

#include "neml2/models/solid_mechanics/TotalStrain.h"
#include "neml2/tensors/SSR4.h"

using vecstr = std::vector<std::string>;

namespace neml2
{
register_NEML2_object(TotalStrain);

OptionSet
TotalStrain::expected_options()
{
  OptionSet options = NewModel::expected_options();
  options.set<LabeledAxisAccessor>("elastic_strain") = {{"state", "internal", "Ee"}};
  options.set<LabeledAxisAccessor>("plastic_strain") = {{"state", "internal", "Ep"}};
  options.set<LabeledAxisAccessor>("total_strain") = vecstr{"state", "E"};
  options.set<bool>("rate_form") = false;
  return options;
}

TotalStrain::TotalStrain(const OptionSet & options)
  : NewModel(options),
    _rate_form(options.get<bool>("rate_form")),
    _Ee(declare_input_variable<SR2>(
        options.get<LabeledAxisAccessor>("elastic_strain").with_suffix(_rate_form ? "_rate" : ""))),
    _Ep(declare_input_variable<SR2>(
        options.get<LabeledAxisAccessor>("plastic_strain").with_suffix(_rate_form ? "_rate" : ""))),
    _E(declare_output_variable<SR2>(
        options.get<LabeledAxisAccessor>("total_strain").with_suffix(_rate_form ? "_rate" : "")))
{
}

void
TotalStrain::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _E = _Ee + _Ep;

  if (dout_din)
  {
    auto I = SR2::identity_map(options());
    _E.d(_Ee) = I;
    _E.d(_Ep) = I;
  }

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
