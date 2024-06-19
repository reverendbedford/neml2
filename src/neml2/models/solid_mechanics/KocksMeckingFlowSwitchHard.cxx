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

#include "neml2/models/solid_mechanics/KocksMeckingFlowSwitchHard.h"

namespace neml2
{
register_NEML2_object(KocksMeckingFlowSwitchHard);

OptionSet
KocksMeckingFlowSwitchHard::expected_options()
{
  OptionSet options = Model::expected_options();

  options.set<CrossRef<Scalar>>("g0");

  options.set<VariableName>("activation_energy") = VariableName("forces", "g");

  options.set<VariableName>("rate_independent_flow_rate") =
      VariableName("state", "internal", "ri_rate");
  options.set<VariableName>("rate_dependent_flow_rate") =
      VariableName("state", "internal", "rd_rate");

  options.set<VariableName>("flow_rate") = VariableName("state", "internal", "gamma_rate");
  return options;
}

KocksMeckingFlowSwitchHard::KocksMeckingFlowSwitchHard(const OptionSet & options)
  : Model(options),
    _g0(declare_parameter<Scalar>("g0", "g0")),
    _g(declare_input_variable<Scalar>("activation_energy")),
    _ri_flow(declare_input_variable<Scalar>("rate_independent_flow_rate")),
    _rd_flow(declare_input_variable<Scalar>("rate_dependent_flow_rate")),
    _gamma_dot(declare_output_variable<Scalar>("flow_rate"))
{
}

void
KocksMeckingFlowSwitchHard::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert(!d2out_din2, "Second derivatives not implemented");

  if (out)
  {
    _gamma_dot = Scalar(torch::where(_g < _g0, _ri_flow, _rd_flow));
  }
  if (dout_din)
  {
    _gamma_dot.d(_ri_flow) = Scalar(torch::where(_g < _g0, 1.0, 0.0));
    _gamma_dot.d(_rd_flow) = Scalar(torch::where(_g < _g0, 0.0, 1.0));
  }
}

} // namespace neml2
