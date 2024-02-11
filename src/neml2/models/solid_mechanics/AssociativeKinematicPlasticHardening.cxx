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
  options.set<VariableName>("kinematic_hardening_direction") = {{"state", "internal", "NX"}};
  options.set<VariableName>("kinematic_plastic_strain_rate") = {{"state", "internal", "Kp_rate"}};
  return options;
}

AssociativeKinematicPlasticHardening::AssociativeKinematicPlasticHardening(
    const OptionSet & options)
  : FlowRule(options),
    _NX(declare_input_variable<SR2>(options.get<VariableName>("kinematic_hardening_direction"))),
    _Kp_dot(
        declare_output_variable<SR2>(options.get<VariableName>("kinematic_plastic_strain_rate")))
{
}

void
AssociativeKinematicPlasticHardening::set_value(bool out, bool dout_din, bool d2out_din2)
{
  // For associative flow,
  // Kp_dot = - gamma_dot * NX
  //     NX = df/dX

  if (out)
    _Kp_dot = -_gamma_dot * _NX;

  if (dout_din || d2out_din2)
  {
    auto I = SR2::identity_map(options());

    if (dout_din)
    {
      _Kp_dot.d(_gamma_dot) = -_NX;
      _Kp_dot.d(_NX) = -_gamma_dot * I;
    }

    if (d2out_din2)
    {
      // I don't know when this will be useful, but since it's easy...
      _Kp_dot.d(_gamma_dot, _NX) = -I;
      _Kp_dot.d(_NX, _gamma_dot) = -I;
    }
  }
}
} // namespace neml2
