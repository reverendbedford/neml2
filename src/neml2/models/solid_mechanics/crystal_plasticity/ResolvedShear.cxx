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

#include "neml2/models/solid_mechanics/crystal_plasticity/ResolvedShear.h"
#include "neml2/models/crystallography/CrystalGeometry.h"

#include "neml2/tensors/tensors.h"
#include "neml2/tensors/list_tensors.h"

using vecstr = std::vector<std::string>;

namespace neml2
{
register_NEML2_object(ResolvedShear);

OptionSet
ResolvedShear::expected_options()
{
  OptionSet options = Model::expected_options();

  options.set<LabeledAxisAccessor>("resolved_shears") =
      vecstr{"state", "internal", "resolved_shears"};
  options.set<LabeledAxisAccessor>("stress") = vecstr{"state", "internal", "cauchy_stress"};
  options.set<LabeledAxisAccessor>("orientation") = vecstr{"state", "orientation"};

  options.set<std::string>("crystal_geometry_name") = "crystal_geometry";

  return options;
}

ResolvedShear::ResolvedShear(const OptionSet & options)
  : Model(options),
    crystal_geometry(include_data<crystallography::CrystalGeometry>(
        options.get<std::string>("crystal_geometry_name"))),
    resolved_shears(declare_output_variable_list<Scalar>(
        options.get<LabeledAxisAccessor>("resolved_shears"), crystal_geometry.nslip())),
    stress(declare_input_variable<SR2>(options.get<LabeledAxisAccessor>("stress"))),
    orientation(declare_input_variable<Rot>(options.get<LabeledAxisAccessor>("orientation")))
{
  setup();
}

void
ResolvedShear::set_value(const LabeledVector & in,
                         LabeledVector * out,
                         LabeledMatrix * dout_din,
                         LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");
  // Grab the input
  const auto S = in.get<SR2>(stress);
  const auto R = in.get<Rot>(orientation);

  if (out)
    out->set_list(crystal_geometry.M().rotate(R.list_unsqueeze()).inner(S.list_unsqueeze()),
                  resolved_shears);

  if (dout_din)
  {
    dout_din->set(BatchTensor(crystal_geometry.M().rotate(R.list_unsqueeze()), S.batch_dim()),
                  resolved_shears,
                  stress);
    auto deriv = SR2(BatchTensor(crystal_geometry.M().drotate(R.list_unsqueeze()), R.batch_dim())
                         .base_transpose(-1, -2),
                     R.batch_dim() + 2)
                     .inner(S.batch_unsqueeze(-1).batch_unsqueeze(-1));
    dout_din->set(BatchTensor(deriv, R.batch_dim()), resolved_shears, orientation);
  }
}

} // namespace neml2
