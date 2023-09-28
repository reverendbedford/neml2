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

#include "neml2/models/solid_mechanics/AssociativeKinematicPlasticHardening.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(AssociativeKinematicPlasticHardening);

OptionSet
AssociativeKinematicPlasticHardening::expected_options()
{
  OptionSet options = FlowRule::expected_options();
  options.set<LabeledAxisAccessor>("kinematic_hardening_direction") = {{"state", "internal", "NX"}};
  options.set<LabeledAxisAccessor>("kinematic_plastic_strain_rate") = {
      {"state", "internal", "Kp_rate"}};
  return options;
}

AssociativeKinematicPlasticHardening::AssociativeKinematicPlasticHardening(
    const OptionSet & options)
  : FlowRule(options),
    kinematic_hardening_direction(declare_input_variable<SR2>(
        options.get<LabeledAxisAccessor>("kinematic_hardening_direction"))),
    kinematic_plastic_strain_rate(declare_output_variable<SR2>(
        options.get<LabeledAxisAccessor>("kinematic_plastic_strain_rate")))
{
  setup();
}

void
AssociativeKinematicPlasticHardening::set_value(const LabeledVector & in,
                                                LabeledVector * out,
                                                LabeledMatrix * dout_din,
                                                LabeledTensor3D * d2out_din2) const
{
  const auto options = in.options();

  // For associative flow,
  // Kp_dot = - gamma_dot * NX
  //     NX = df/dX
  const auto gamma_dot = in.get<Scalar>(flow_rate);
  const auto NX = in.get<SR2>(kinematic_hardening_direction);

  if (out)
    out->set(-gamma_dot * NX, kinematic_plastic_strain_rate);

  if (dout_din || d2out_din2)
  {
    auto I = SR2::identity_map(options);

    if (dout_din)
    {
      dout_din->set(-NX, kinematic_plastic_strain_rate, flow_rate);
      dout_din->set(-gamma_dot * I, kinematic_plastic_strain_rate, kinematic_hardening_direction);
    }

    if (d2out_din2)
    {
      // I don't know when this will be useful, but since it's easy...
      d2out_din2->set(-I, kinematic_plastic_strain_rate, flow_rate, kinematic_hardening_direction);
      d2out_din2->set(-I, kinematic_plastic_strain_rate, kinematic_hardening_direction, flow_rate);
    }
  }
}
} // namespace neml2
