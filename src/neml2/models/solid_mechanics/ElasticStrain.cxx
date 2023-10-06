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

#include "neml2/models/solid_mechanics/ElasticStrain.h"
#include "neml2/tensors/SSR4.h"

using vecstr = std::vector<std::string>;

namespace neml2
{
register_NEML2_object(ElasticStrain);

OptionSet
ElasticStrain::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<LabeledAxisAccessor>("total_strain") = vecstr{"forces", "E"};
  options.set<LabeledAxisAccessor>("plastic_strain") = {{"state", "internal", "Ep"}};
  options.set<LabeledAxisAccessor>("elastic_strain") = {{"state", "internal", "Ee"}};
  options.set<bool>("rate_form") = false;
  return options;
}

ElasticStrain::ElasticStrain(const OptionSet & options)
  : Model(options),
    _rate_form(options.get<bool>("rate_form")),
    total_strain(declare_input_variable<SR2>(
        options.get<LabeledAxisAccessor>("total_strain").with_suffix(_rate_form ? "_rate" : ""))),
    plastic_strain(declare_input_variable<SR2>(
        options.get<LabeledAxisAccessor>("plastic_strain").with_suffix(_rate_form ? "_rate" : ""))),
    elastic_strain(declare_output_variable<SR2>(
        options.get<LabeledAxisAccessor>("elastic_strain").with_suffix(_rate_form ? "_rate" : "")))
{
  setup();
}

void
ElasticStrain::set_value(const LabeledVector & in,
                         LabeledVector * out,
                         LabeledMatrix * dout_din,
                         LabeledTensor3D * d2out_din2) const
{
  // Simple additive decomposition:
  // elastic strain = total strain - plastic strain
  if (out)
  {
    auto E = in.get<SR2>(total_strain);
    auto Ep = in.get<SR2>(plastic_strain);
    out->set(E - Ep, elastic_strain);
  }

  if (dout_din)
  {
    auto I = SR2::identity_map(in.options());
    dout_din->set(I, elastic_strain, total_strain);
    dout_din->set(-I, elastic_strain, plastic_strain);
  }

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
