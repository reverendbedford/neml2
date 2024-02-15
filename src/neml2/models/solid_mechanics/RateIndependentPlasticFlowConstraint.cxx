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

OptionSet
RateIndependentPlasticFlowConstraint::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<VariableName>("yield_function") = VariableName("state", "internal", "fp");
  options.set<VariableName>("flow_rate") = VariableName("state", "internal", "gamma_rate");
  options.set<Real>("yielding_tolerance") = 1e-8;
  return options;
}

RateIndependentPlasticFlowConstraint::RateIndependentPlasticFlowConstraint(
    const OptionSet & options)
  : Model(options),
    _ytol(options.get<Real>("yielding_tolerance")),
    _fp(declare_input_variable<Scalar>("yield_function")),
    _gamma_dot(declare_input_variable<Scalar>("flow_rate")),
    _r(declare_output_variable<Scalar>(
        options.get<VariableName>("flow_rate").slice(1).on("residual")))
{
}

void
RateIndependentPlasticFlowConstraint::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  // The residual is the yield function itself when the stress state is "outside" the yield surface,
  // also called return mapping. The residual is the hardening rate when the stress state is
  // "inside" the yield surface, as the hardening rate is by definition zero.
  const auto yielding = _fp >= -_ytol;
  const auto not_yielding = torch::logical_not(yielding);

  if (out)
  {
    auto r = Scalar::empty_like(_r);
    r.index_put_({not_yielding}, Scalar(_gamma_dot).index({not_yielding}));
    r.index_put_({yielding}, Scalar(_fp).index({yielding}));
    _r = r;
  }

  if (dout_din)
  {
    auto dr_dgamma_dot = Scalar::empty_like(_r);
    dr_dgamma_dot.index_put_({not_yielding}, 1);
    dr_dgamma_dot.index_put_({yielding}, 0);
    _r.d(_gamma_dot) = dr_dgamma_dot;

    auto dr_dfp = Scalar::empty_like(_r);
    dr_dfp.index_put_({not_yielding}, 0);
    dr_dfp.index_put_({yielding}, 1);
    _r.d(_fp) = dr_dfp;
  }
}

} // namespace neml2
