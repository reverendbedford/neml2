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

#include "neml2/models/solid_mechanics/RateIndependentPlasticFlowConstraint.h"

namespace neml2
{
register_NEML2_object(RateIndependentPlasticFlowConstraint);

ParameterSet
RateIndependentPlasticFlowConstraint::expected_params()
{
  ParameterSet params = Model::expected_params();
  return params;
}

RateIndependentPlasticFlowConstraint::RateIndependentPlasticFlowConstraint(
    const ParameterSet & params)
  : Model(params),
    hardening_rate(declareInputVariable<Scalar>({"state", "hardening_rate"})),
    yield_function(declareInputVariable<Scalar>({"state", "yield_function"})),
    consistency_condition(declareOutputVariable<Scalar>({"residual", "consistency_condition"}))
{
  setup();
}

void
RateIndependentPlasticFlowConstraint::set_value(LabeledVector in,
                                                LabeledVector out,
                                                LabeledMatrix * dout_din) const
{
  // Grab the yield function
  auto gamma_dot = in.get<Scalar>(hardening_rate);
  auto f = in.get<Scalar>(yield_function);
  out.set(gamma_dot * f, consistency_condition);

  if (dout_din)
  {
    dout_din->set(f, consistency_condition, hardening_rate);
    dout_din->set(gamma_dot, consistency_condition, yield_function);
  }
}

} // namespace neml2
