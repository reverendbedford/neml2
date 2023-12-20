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

#include "neml2/models/solid_mechanics/crystal_plasticity/VoceSingleSlipHardeningRule.h"

using vecstr = std::vector<std::string>;

namespace neml2
{
register_NEML2_object(VoceSingleSlipHardeningRule);

OptionSet
VoceSingleSlipHardeningRule::expected_options()
{
  OptionSet options = SingleSlipHardeningRule::expected_options();

  options.set<CrossRef<Scalar>>("initial_slope");
  options.set<CrossRef<Scalar>>("saturated_hardening");

  return options;
}

VoceSingleSlipHardeningRule::VoceSingleSlipHardeningRule(const OptionSet & options)
  : SingleSlipHardeningRule(options),
    _theta_0(declare_parameter<Scalar>("initial_slope", "initial_slope")),
    _tau_f(declare_parameter<Scalar>("saturated_hardening", "saturated_hardening"))
{
  setup();
}

void
VoceSingleSlipHardeningRule::set_value(const LabeledVector & in,
                                       LabeledVector * out,
                                       LabeledMatrix * dout_din,
                                       LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  // Grab the input
  const auto sg = in.get<Scalar>(sum_slip_rates);
  const auto tau = in.get<Scalar>(slip_hardening);

  if (out)
    out->set(_theta_0 * (1 - tau / _tau_f) * sg, slip_hardening_rate);

  if (dout_din)
  {
    dout_din->set(-_theta_0 / _tau_f * sg, slip_hardening_rate, slip_hardening);
    dout_din->set(_theta_0 * (1 - tau / _tau_f), slip_hardening_rate, sum_slip_rates);
  }
}
} // namespace neml2
