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
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(PowerLawSlipRule);

OptionSet
PowerLawSlipRule::expected_options()
{
  OptionSet options = SlipRule::expected_options();
  options.doc() =
      "Power law slip rule defined as \\f$ \\dot{\\gamma}_i = \\dot{\\gamma}_0 \\left| "
      "\\frac{\\tau_i}{\\hat{\\tau}_i} \\right|^{n-1} \\frac{\\tau_i}{\\hat{\\tau}_i} \\f$ with "
      "\\f$ \\dot{\\gamma}_i \\f$ the slip rate on system \\f$ i \\f$, \\f$ \\tau_i \\f$ the "
      "resolved shear, \\f$ \\hat{\\tau}_i \\f$ the slip system strength, \\f$ n \\f$ the rate "
      "senstivity, and \\f$ \\dot{\\gamma}_0 \\f$ a reference slip rate.";

  options.set_parameter<CrossRef<Scalar>>("gamma0");
  options.set("gamma0").doc() = "Reference slip rate";

  options.set_parameter<CrossRef<Scalar>>("n");
  options.set("n").doc() = "Rate sensitivity exponent";

  return options;
}

PowerLawSlipRule::PowerLawSlipRule(const OptionSet & options)
  : SlipRule(options),
    _gamma0(declare_parameter<Scalar>("gamma0", "gamma0")),
    _n(declare_parameter<Scalar>("n", "n"))
{
}

void
PowerLawSlipRule::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  // Grab the input
  const auto rss = Scalar(_rss, batch_dim() + 1);
  const auto tau = Scalar(_tau, batch_dim() + 1);

  if (out)
    _g = Tensor(_gamma0 * math::pow(abs(rss / tau), _n - 1.0) * rss / tau, batch_dim());

  if (dout_din)
  {
    if (_rss.is_dependent())
      _g.d(_rss) =
          Tensor(math::batch_diag_embed(_gamma0 * _n * math::pow(abs(rss / tau), _n - 1.0) / tau),
                 batch_dim());

    if (_tau.is_dependent())
      _g.d(_tau) =
          Tensor(math::batch_diag_embed(-_n * _gamma0 * rss * math::pow(abs(rss), _n - 1.0) /
                                        math::pow(tau, _n + 1)),
                 batch_dim());
  }
}
} // namespace neml2
