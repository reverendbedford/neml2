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

#include "neml2/models/solid_mechanics/GursonCavitation.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(GursonCavitation);

OptionSet
GursonCavitation::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<LabeledAxisAccessor>("plastic_strain_rate") = {{"state", "internal", "Ep_rate"}};
  options.set<LabeledAxisAccessor>("void_fraction") = {{"state", "internal", "f"}};
  options.set<LabeledAxisAccessor>("void_fraction_rate") = {{"state", "internal", "f_rate"}};
  return options;
}

GursonCavitation::GursonCavitation(const OptionSet & options)
  : Model(options),
    plastic_strain_rate(
        declare_input_variable<SR2>(options.get<LabeledAxisAccessor>("plastic_strain_rate"))),
    void_fraction(
        declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("void_fraction"))),
    void_fraction_rate(
        declare_output_variable<Scalar>(options.get<LabeledAxisAccessor>("void_fraction_rate")))
{
  setup();
}

void
GursonCavitation::set_value(const LabeledVector & in,
                            LabeledVector * out,
                            LabeledMatrix * dout_din,
                            LabeledTensor3D * d2out_din2) const
{
  const auto options = in.options();

  auto f = in.get<Scalar>(void_fraction);
  auto tr_ep = in.get<SR2>(plastic_strain_rate).tr();

  if (out)
    out->set((1 - f) * tr_ep, void_fraction_rate);

  if (dout_din || d2out_din2)
  {
    auto I = SR2::identity(options);
    if (dout_din)
    {
      dout_din->set(-tr_ep, void_fraction_rate, void_fraction);
      dout_din->set(I * (1 - f), void_fraction_rate, plastic_strain_rate);
    }
    // No idea if this will ever be used, but why not as it's easy?
    if (d2out_din2)
    {
      d2out_din2->set(-I, void_fraction_rate, void_fraction, plastic_strain_rate);
      d2out_din2->set(-I, void_fraction_rate, plastic_strain_rate, void_fraction);
    }
  }
}
} // namespace neml2
