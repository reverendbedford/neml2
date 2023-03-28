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
    yield_function(declareInputVariable<Scalar>({"state", "yield_function"})),
    hardening_rate(declareInputVariable<Scalar>({"state", "hardening_rate"})),
    consistency_condition(declareOutputVariable<Scalar>({"residual", "consistency_condition"}))
{
  setup();
}

void
RateIndependentPlasticFlowConstraint::set_value(LabeledVector in,
                                                LabeledVector out,
                                                LabeledMatrix * dout_din) const
{
  auto f = in.get<Scalar>(yield_function);
  auto gamma_dot = in.get<Scalar>(hardening_rate);

  // The residual is the yield function itself when the stress state is "outside" the yield surface,
  // also called return mapping. The residual is the hardening rate when the stress state is
  // "inside" the yield surface, as the hardening rate is by definition zero.
  Scalar r(0, in.batch_size());
  r.index_put_({f < 0}, gamma_dot.index({f < 0}));
  r.index_put_({f >= 0}, f.index({f >= 0}));

  out.set(r, consistency_condition);

  if (dout_din)
  {
    Scalar dr_dgamma_dot(0, in.batch_size());
    dr_dgamma_dot.index_put_({f < 0}, 1);

    Scalar dr_df(0, in.batch_size());
    dr_df.index_put_({f >= 0}, 1);

    dout_din->set(dr_dgamma_dot, consistency_condition, hardening_rate);
    dout_din->set(dr_df, consistency_condition, yield_function);
  }
}

} // namespace neml2
