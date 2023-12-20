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

#include "neml2/models/solid_mechanics/crystal_plasticity/ElasticStrainRate.h"

#include "neml2/tensors/tensors.h"
#include "neml2/misc/math.h"

using vecstr = std::vector<std::string>;

namespace neml2
{
register_NEML2_object(ElasticStrainRate);

OptionSet
ElasticStrainRate::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<LabeledAxisAccessor>("elastic_strain_rate") = vecstr{"state", "elastic_strain_rate"};
  options.set<LabeledAxisAccessor>("elastic_strain") = vecstr{"state", "elastic_strain"};

  options.set<LabeledAxisAccessor>("deformation_rate") = vecstr{"forces", "deformation_rate"};
  options.set<LabeledAxisAccessor>("vorticity") = vecstr{"forces", "vorticity"};

  options.set<LabeledAxisAccessor>("plastic_deformation_rate") =
      vecstr{"state", "internal", "plastic_deformation_rate"};
  return options;
}

ElasticStrainRate::ElasticStrainRate(const OptionSet & options)
  : Model(options),
    elastic_strain_rate(
        declare_output_variable<SR2>(options.get<LabeledAxisAccessor>("elastic_strain_rate"))),
    elastic_strain(declare_input_variable<SR2>(options.get<LabeledAxisAccessor>("elastic_strain"))),
    deformation_rate(
        declare_input_variable<SR2>(options.get<LabeledAxisAccessor>("deformation_rate"))),
    vorticity(declare_input_variable<WR2>(options.get<LabeledAxisAccessor>("vorticity"))),
    plastic_deformation_rate(
        declare_input_variable<SR2>(options.get<LabeledAxisAccessor>("plastic_deformation_rate")))
{
  setup();
}

void
ElasticStrainRate::set_value(const LabeledVector & in,
                             LabeledVector * out,
                             LabeledMatrix * dout_din,
                             LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  // Grab the input
  const auto e = in.get<SR2>(elastic_strain);
  const auto d = in.get<SR2>(deformation_rate);
  const auto w = in.get<WR2>(vorticity);
  const auto dp = in.get<SR2>(plastic_deformation_rate);

  if (out)
    out->set(d - dp + math::skew_and_sym_to_sym(e, w), elastic_strain_rate);

  if (dout_din)
  {
    auto I = SSR4::identity_sym(e.dtype());
    dout_din->set(math::d_skew_and_sym_to_sym_d_sym(w), elastic_strain_rate, elastic_strain);
    dout_din->set(I, elastic_strain_rate, deformation_rate);
    dout_din->set(math::d_skew_and_sym_to_sym_d_skew(e), elastic_strain_rate, vorticity);
    dout_din->set(-I, elastic_strain_rate, plastic_deformation_rate);
  }
}
} // namespace neml2