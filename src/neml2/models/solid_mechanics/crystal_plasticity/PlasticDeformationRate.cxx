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

#include "neml2/models/solid_mechanics/crystal_plasticity/PlasticDeformationRate.h"
#include "neml2/models/crystallography/CrystalGeometry.h"

#include "neml2/tensors/tensors.h"
#include "neml2/tensors/list_tensors.h"

using vecstr = std::vector<std::string>;

namespace neml2
{
register_NEML2_object(PlasticDeformationRate);

OptionSet
PlasticDeformationRate::expected_options()
{
  OptionSet options = NewModel::expected_options();

  options.set<LabeledAxisAccessor>("plastic_deformation_rate") =
      vecstr{"state", "internal", "plastic_deformation_rate"};

  options.set<LabeledAxisAccessor>("orientation") = vecstr{"state", "orientation"};

  options.set<LabeledAxisAccessor>("slip_rates") = vecstr{"state", "internal", "slip_rates"};

  options.set<std::string>("crystal_geometry_name") = "crystal_geometry";

  return options;
}

PlasticDeformationRate::PlasticDeformationRate(const OptionSet & options)
  : NewModel(options),
    _crystal_geometry(register_data<crystallography::CrystalGeometry>(
        options.get<std::string>("crystal_geometry_name"))),
    _dp(declare_output_variable<SR2>(options.get<LabeledAxisAccessor>("plastic_deformation_rate"))),
    _R(declare_input_variable<Rot>(options.get<LabeledAxisAccessor>("orientation"))),
    _g(declare_input_variable_list<Scalar>(options.get<LabeledAxisAccessor>("slip_rates"),
                                           _crystal_geometry.nslip()))
{
}

void
PlasticDeformationRate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  // Grab the input
  const auto g = Scalar(_g, batch_dim() + 1);
  const auto dp_crystal = (g * _crystal_geometry.M()).list_sum();

  if (out)
    _dp = dp_crystal.rotate(Rot(_R));

  if (dout_din)
  {
    _dp.d(_g) = list_derivative_outer_product_b(
        [](auto a, auto b) { return b.rotate(a); }, Rot(_R), _crystal_geometry.M());
    _dp.d(_R) = dp_crystal.drotate(Rot(_R));
  }
}
} // namespace neml2
