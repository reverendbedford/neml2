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

#include "neml2/models/WR2ExplicitExponentialTimeIntegration.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
register_NEML2_object(WR2ExplicitExponentialTimeIntegration);

OptionSet
WR2ExplicitExponentialTimeIntegration::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<VariableName>("variable");
  options.set<VariableName>("time") = VariableName("t");
  return options;
}

WR2ExplicitExponentialTimeIntegration::WR2ExplicitExponentialTimeIntegration(
    const OptionSet & options)
  : Model(options),
    _var_name(options.get<VariableName>("variable")),
    _var_rate_name(_var_name.with_suffix("_rate")),
    _s(declare_output_variable<Rot>(_var_name.on("state"))),
    _s_dot(declare_input_variable<WR2>(_var_rate_name.on("state"))),
    _sn(declare_input_variable<Rot>(_var_name.on("old_state"))),
    _t(declare_input_variable<Scalar>(options.get<VariableName>("time").on("forces"))),
    _tn(declare_input_variable<Scalar>(options.get<VariableName>("time").on("old_forces")))
{
}

void
WR2ExplicitExponentialTimeIntegration::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  const auto dt = _t - _tn;

  // Incremental rotation
  const auto inc = (_s_dot * dt).exp();

  if (out)
    _s = Rot(_sn).rotate(inc);

  if (dout_din)
  {
    const auto de = (_s_dot * dt).dexp();
    _s.d(_s_dot) = Rot(_sn).drotate(inc) * de * dt;
    if (Model::stage == Model::Stage::UPDATING)
    {
      _s.d(_sn) = Rot(_sn).drotate_self(inc);
      _s.d(_t) = Rot(_sn).drotate(inc) * de * Vec(_s_dot.value());
      _s.d(_tn) = -Rot(_sn).drotate(inc) * de * Vec(_s_dot.value());
    }
  }
}

} // namespace neml2
