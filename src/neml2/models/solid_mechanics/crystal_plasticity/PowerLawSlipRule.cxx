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

#include "neml2/models/solid_mechanics/crystal_plasticity/PowerLawSlipRule.h"
#include "neml2/models/solid_mechanics/crystal_plasticity/SlipRule.h"

namespace neml2
{
register_NEML2_object(PowerLawSlipRule);

OptionSet
PowerLawSlipRule::expected_options()
{
  OptionSet options = SlipRule::expected_options();

  options.set<CrossRef<Scalar>>("gamma0");
  options.set<CrossRef<Scalar>>("n");

  return options;
}

PowerLawSlipRule::PowerLawSlipRule(const OptionSet & options)
  : SlipRule(options),
    _gamma0(declare_parameter<Scalar>("gamma0", "gamma0")),
    _n(declare_parameter<Scalar>("n", "n"))
{
  setup();
}

void
PowerLawSlipRule::set_value(const LabeledVector & in,
                            LabeledVector * out,
                            LabeledMatrix * dout_din,
                            LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");
  // Grab the input
  const auto tau = in.get_list<Scalar>(resolved_shears);
  const auto tau_bar = in.get_list<Scalar>(slip_strengths);

  if (out)
    out->set_list(_gamma0 * math::pow(abs(tau / tau_bar), _n - 1.0) * tau / tau_bar, slip_rates);

  if (dout_din)
  {
    dout_din->set_list(
        math::batch_diag_embed(_gamma0 * _n * math::pow(abs(tau / tau_bar), _n - 1.0) / tau_bar),
        slip_rates,
        resolved_shears);
    dout_din->set_list(math::batch_diag_embed(-_n * _gamma0 * tau * math::pow(abs(tau), _n - 1.0) /
                                              math::pow(tau_bar, _n + 1)),
                       slip_rates,
                       slip_strengths);
  }
}
} // namespace neml2
