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

using namespace std::literals;

namespace neml2
{
register_NEML2_object(RateIndependentPlasticFlowConstraint);

OptionSet
RateIndependentPlasticFlowConstraint::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<LabeledAxisAccessor>("yield_function") = {{"state", "internal", "fp"}};
  options.set<LabeledAxisAccessor>("flow_rate") = {{"state"s, "gamma_rate"s}};
  return options;
}

RateIndependentPlasticFlowConstraint::RateIndependentPlasticFlowConstraint(
    const OptionSet & options)
  : Model(options),
    yield_function(
        declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("yield_function"))),
    flow_rate(declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("flow_rate"))),
    consistency_condition(declare_output_variable<Scalar>(
        options.get<LabeledAxisAccessor>("flow_rate").slice(1).on("residual")))
{
  setup();
}

void
RateIndependentPlasticFlowConstraint::set_value(const LabeledVector & in,
                                                LabeledVector * out,
                                                LabeledMatrix * dout_din,
                                                LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  const auto options = in.options();
  const auto batch_sz = in.batch_sizes();

  auto f = in.get<Scalar>(yield_function);
  auto gamma_dot = in.get<Scalar>(flow_rate);

  // The residual is the yield function itself when the stress state is "outside" the yield surface,
  // also called return mapping. The residual is the hardening rate when the stress state is
  // "inside" the yield surface, as the hardening rate is by definition zero.
  auto r = Scalar::empty(batch_sz, options);
  r.index_put_({f < -TOL2}, gamma_dot.index({f < -TOL2}));
  r.index_put_({f >= -TOL2}, f.index({f >= -TOL2}));

  if (out)
    out->set(r, consistency_condition);

  if (dout_din)
  {
    auto dr_dgamma_dot = Scalar::empty(batch_sz, options);
    dr_dgamma_dot.index_put_({f < -TOL2}, 1);
    dr_dgamma_dot.index_put_({f >= -TOL2}, 0);

    auto dr_df = Scalar::empty(batch_sz, options);
    dr_df.index_put_({f < -TOL2}, 0);
    dr_df.index_put_({f >= -TOL2}, 1);

    dout_din->set(dr_dgamma_dot, consistency_condition, flow_rate);
    dout_din->set(dr_df, consistency_condition, yield_function);
  }
}

} // namespace neml2
