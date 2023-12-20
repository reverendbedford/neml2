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
  options.set<LabeledAxisAccessor>("variable");
  options.set<LabeledAxisAccessor>("time") = {{"t"}};
  return options;
}

WR2ExplicitExponentialTimeIntegration::WR2ExplicitExponentialTimeIntegration(
    const OptionSet & options)
  : Model(options),
    _var_name(options.get<LabeledAxisAccessor>("variable")),
    _var_rate_name(_var_name.with_suffix("_rate")),
    var_rate(declare_input_variable<WR2>(_var_rate_name.on("state"))),
    var(declare_output_variable<Rot>(_var_name.on("state"))),
    var_n(declare_input_variable<Rot>(_var_name.on("old_state"))),
    time(declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("time").on("forces"))),
    time_n(
        declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("time").on("old_forces")))
{
  this->setup();
}

void
WR2ExplicitExponentialTimeIntegration::set_value(const LabeledVector & in,
                                                 LabeledVector * out,
                                                 LabeledMatrix * dout_din,
                                                 LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  auto s_n = in.get<Rot>(var_n);
  auto t_np1 = in.get<Scalar>(time);
  auto t_n = in.get<Scalar>(time_n);
  auto s_dot = in.get<WR2>(var_rate);
  auto dt = t_np1 - t_n;

  // Incremental rotation
  auto inc = (s_dot * dt).exp();

  if (out)
    out->set(s_n.rotate(inc), var);

  if (dout_din)
  {
    auto de = (s_dot * dt).dexp();

    dout_din->set(s_n.drotate(inc) * de * dt, var, var_rate);
    if (Model::stage == Model::Stage::UPDATING)
    {
      dout_din->set(s_n.drotate_self(inc), var, var_n);
      dout_din->set(s_n.drotate(inc) * de * Vec(s_dot), var, time);
      dout_din->set(-s_n.drotate(inc) * de * Vec(s_dot), var, time_n);
    }
  }
}

} // namespace neml2
