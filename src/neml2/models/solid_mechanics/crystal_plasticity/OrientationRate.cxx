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

#include "neml2/models/solid_mechanics/crystal_plasticity/OrientationRate.h"

#include "neml2/tensors/tensors.h"

using vecstr = std::vector<std::string>;

namespace neml2
{
register_NEML2_object(OrientationRate);

OptionSet
OrientationRate::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<LabeledAxisAccessor>("orientation_rate") = vecstr{"state", "orientation_rate"};
  options.set<LabeledAxisAccessor>("elastic_strain") = vecstr{"state", "elastic_strain"};

  options.set<LabeledAxisAccessor>("vorticity") = vecstr{"state", "vorticity"};

  options.set<LabeledAxisAccessor>("plastic_deformation_rate") =
      vecstr{"state", "internal", "plastic_deformation_rate"};
  options.set<LabeledAxisAccessor>("plastic_vorticity") =
      vecstr{"state", "internal", "plastic_vorticity"};
  return options;
}

OrientationRate::OrientationRate(const OptionSet & options)
  : Model(options),
    orientation_rate(
        declare_output_variable<WR2>(options.get<LabeledAxisAccessor>("orientation_rate"))),
    elastic_strain(declare_input_variable<SR2>(options.get<LabeledAxisAccessor>("elastic_strain"))),
    vorticity(declare_input_variable<WR2>(options.get<LabeledAxisAccessor>("vorticity"))),
    plastic_deformation_rate(
        declare_input_variable<SR2>(options.get<LabeledAxisAccessor>("plastic_deformation_rate"))),
    plastic_vorticity(
        declare_input_variable<WR2>(options.get<LabeledAxisAccessor>("plastic_vorticity")))
{
  setup();
}

void
OrientationRate::set_value(const LabeledVector & in,
                           LabeledVector * out,
                           LabeledMatrix * dout_din,
                           LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  // Grab the input
  const auto e = in.get<SR2>(elastic_strain);
  const auto w = in.get<WR2>(vorticity);
  const auto dp = in.get<SR2>(plastic_deformation_rate);
  const auto wp = in.get<WR2>(plastic_vorticity);

  if (out)
    out->set(w - wp + product_abmba(dp, e), orientation_rate);

  if (dout_din)
  {
    auto I = WWR4::identity(e.dtype());
    dout_din->set(d_product_abmba_db(dp), orientation_rate, elastic_strain);
    dout_din->set(I, orientation_rate, vorticity);
    dout_din->set(d_product_abmba_da(e), orientation_rate, plastic_deformation_rate);
    dout_din->set(-I, orientation_rate, plastic_vorticity);
  }
}
} // namespace neml2