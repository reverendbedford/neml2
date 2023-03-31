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

#include "neml2/models/solid_mechanics/AssociativeIsotropicPlasticHardening.h"

namespace neml2
{
register_NEML2_object(AssociativeIsotropicPlasticHardening);

ParameterSet
AssociativeIsotropicPlasticHardening::expected_params()
{
  ParameterSet params = FlowRule::expected_params();
  params.set<LabeledAxisAccessor>("isotropic_hardening_direction") = {{"state", "internal", "Nk"}};
  params.set<LabeledAxisAccessor>("equivalent_plastic_strain_rate") = {
      {"state", "internal", "ep_rate"}};
  return params;
}

AssociativeIsotropicPlasticHardening::AssociativeIsotropicPlasticHardening(
    const ParameterSet & params)
  : FlowRule(params),
    isotropic_hardening_direction(declare_input_variable<Scalar>(
        params.get<LabeledAxisAccessor>("isotropic_hardening_direction"))),
    equivalent_plastic_strain_rate(declare_output_variable<Scalar>(
        params.get<LabeledAxisAccessor>("equivalent_plastic_strain_rate")))
{
  setup();
}

void
AssociativeIsotropicPlasticHardening::set_value(LabeledVector in,
                                                LabeledVector * out,
                                                LabeledMatrix * dout_din,
                                                LabeledTensor3D * d2out_din2) const
{
  // For associative flow,
  // ep_dot = - gamma_dot * Nk
  //     Nk = df/dk
  auto gamma_dot = in.get<Scalar>(flow_rate);
  auto Nk = in.get<Scalar>(isotropic_hardening_direction);

  if (out)
    out->set(-gamma_dot * Nk, equivalent_plastic_strain_rate);

  if (dout_din)
  {
    dout_din->set(-Nk, equivalent_plastic_strain_rate, flow_rate);
    dout_din->set(-gamma_dot, equivalent_plastic_strain_rate, isotropic_hardening_direction);
  }

  if (d2out_din2)
  {
    auto I = Scalar::identity_map(in.options());
    // I don't know when this will be useful, but since it's easy...
    d2out_din2->set(-I, equivalent_plastic_strain_rate, flow_rate, isotropic_hardening_direction);
    d2out_din2->set(-I, equivalent_plastic_strain_rate, isotropic_hardening_direction, flow_rate);
  }
}
} // namespace neml2
