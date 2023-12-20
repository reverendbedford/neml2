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

#include "neml2/models/solid_mechanics/PerzynaPlasticFlowRate.h"

namespace neml2
{
register_NEML2_object(PerzynaPlasticFlowRate);

OptionSet
PerzynaPlasticFlowRate::expected_options()
{
  OptionSet options = PlasticFlowRate::expected_options();
  options.set<CrossRef<Scalar>>("reference_stress");
  options.set<CrossRef<Scalar>>("exponent");
  return options;
}

PerzynaPlasticFlowRate::PerzynaPlasticFlowRate(const OptionSet & options)
  : PlasticFlowRate(options),
    _eta(declare_parameter<Scalar>("eta", "reference_stress")),
    _n(declare_parameter<Scalar>("n", "exponent"))
{
}

void
PerzynaPlasticFlowRate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  // Compute the Perzyna approximation of the yield surface
  auto Hf = math::heaviside(Scalar(_f));
  auto f_abs = math::abs(Scalar(_f));
  auto gamma_dot_m = math::pow(f_abs / _eta, _n);
  auto gamma_dot = gamma_dot_m * Hf;

  if (out)
    _gamma_dot = gamma_dot;

  if (dout_din || d2out_din2)
  {
    auto dgamma_dot_df = _n / f_abs * gamma_dot;

    if (dout_din)
      _gamma_dot.d(_f) = dgamma_dot_df;

    if (d2out_din2)
      _gamma_dot.d(_f, _f) = (1 - 1 / f_abs) * dgamma_dot_df;
  }
}
} // namespace neml2
