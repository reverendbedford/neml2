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
#include "neml2/misc/math.h"

using vecstr = std::vector<std::string>;

namespace neml2
{
register_NEML2_object(OrientationRate);

OptionSet
OrientationRate::expected_options()
{
  OptionSet options = NewModel::expected_options();
  options.set<LabeledAxisAccessor>("orientation_rate") = vecstr{"state", "orientation_rate"};
  options.set<LabeledAxisAccessor>("elastic_strain") = vecstr{"state", "elastic_strain"};
  options.set<LabeledAxisAccessor>("vorticity") = vecstr{"forces", "vorticity"};
  options.set<LabeledAxisAccessor>("plastic_deformation_rate") =
      vecstr{"state", "internal", "plastic_deformation_rate"};
  options.set<LabeledAxisAccessor>("plastic_vorticity") =
      vecstr{"state", "internal", "plastic_vorticity"};
  return options;
}

OrientationRate::OrientationRate(const OptionSet & options)
  : NewModel(options),
    _R_dot(declare_output_variable<WR2>(options.get<LabeledAxisAccessor>("orientation_rate"))),
    _e(declare_input_variable<SR2>(options.get<LabeledAxisAccessor>("elastic_strain"))),
    _w(declare_input_variable<WR2>(options.get<LabeledAxisAccessor>("vorticity"))),
    _dp(declare_input_variable<SR2>(options.get<LabeledAxisAccessor>("plastic_deformation_rate"))),
    _wp(declare_input_variable<WR2>(options.get<LabeledAxisAccessor>("plastic_vorticity")))
{
}

void
OrientationRate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
    _R_dot = _w - _wp + math::multiply_and_make_skew(SR2(_dp), SR2(_e));

  if (dout_din)
  {
    const auto I = WWR4::identity(options());
    _R_dot.d(_e) = math::d_multiply_and_make_skew_d_second(SR2(_dp));
    _R_dot.d(_w) = I;
    _R_dot.d(_dp) = math::d_multiply_and_make_skew_d_first(SR2(_e));
    _R_dot.d(_wp) = -I;
  }
}
} // namespace neml2
