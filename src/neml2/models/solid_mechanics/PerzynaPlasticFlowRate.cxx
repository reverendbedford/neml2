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

ParameterSet
PerzynaPlasticFlowRate::expected_params()
{
  ParameterSet params = PlasticFlowRate::expected_params();
  params.set<Real>("reference_stress");
  params.set<Real>("exponent");
  return params;
}

PerzynaPlasticFlowRate::PerzynaPlasticFlowRate(const ParameterSet & params)
  : PlasticFlowRate(params),
    _eta(register_parameter("eta", Scalar(params.get<Real>("reference_stress")), false)),
    _n(register_parameter("n", Scalar(params.get<Real>("exponent")), false))
{
}

void
PerzynaPlasticFlowRate::set_value(const LabeledVector & in,
                                  LabeledVector * out,
                                  LabeledMatrix * dout_din,
                                  LabeledTensor3D * d2out_din2) const
{
  // Grab the yield function
  auto f = in.get<Scalar>(yield_function);

  // Compute the Perzyna approximation of the yield surface
  Scalar Hf = math::heaviside(f);
  Scalar f_abs = torch::abs(f);
  Scalar gamma_dot_m = torch::pow(f_abs / _eta, _n);
  Scalar gamma_dot = gamma_dot_m * Hf;

  if (out)
    out->set(gamma_dot, flow_rate);

  if (dout_din || d2out_din2)
  {
    auto dgamma_dot_df = _n / f_abs * gamma_dot;
    if (dout_din)
      dout_din->set(dgamma_dot_df, flow_rate, yield_function);
    if (d2out_din2)
      d2out_din2->set((1 - 1 / f_abs) * dgamma_dot_df, flow_rate, yield_function, yield_function);
  }
}
} // namespace neml2
