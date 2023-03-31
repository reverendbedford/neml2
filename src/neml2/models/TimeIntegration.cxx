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

#include "neml2/models/TimeIntegration.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
register_NEML2_object(ScalarTimeIntegration);
register_NEML2_object(SymR2TimeIntegration);

template <typename T>
ParameterSet
TimeIntegration<T>::expected_params()
{
  ParameterSet params = Model::expected_params();
  params.set<LabeledAxisAccessor>("variable");
  params.set<LabeledAxisAccessor>("time") = {{"t"}};
  return params;
}

template <typename T>
TimeIntegration<T>::TimeIntegration(const ParameterSet & params)
  : Model(params),
    _var_name(params.get<LabeledAxisAccessor>("variable")),
    _var_rate_name(_var_name.with_suffix("_rate")),
    var_rate(declare_input_variable<T>(_var_rate_name.on("state"))),
    var(declare_output_variable<T>(_var_name.on("state"))),
    var_n(declare_input_variable<T>(_var_name.on("old_state"))),
    time(declare_input_variable<Scalar>(params.get<LabeledAxisAccessor>("time").on("forces"))),
    time_n(declare_input_variable<Scalar>(params.get<LabeledAxisAccessor>("time").on("old_forces")))
{
  this->setup();
}

template <typename T>
void
TimeIntegration<T>::set_value(LabeledVector in,
                              LabeledVector * out,
                              LabeledMatrix * dout_din,
                              LabeledTensor3D * d2out_din2) const
{
  const auto options = in.options();

  auto s_n = in.get<T>(var_n);
  auto t_np1 = in.get<Scalar>(time);
  auto t_n = in.get<Scalar>(time_n);
  auto s_dot = in.get<T>(var_rate);
  auto dt = t_np1 - t_n;

  if (out)
    out->set(s_n + s_dot * dt, var);

  if (dout_din || d2out_din2)
  {
    auto I = T::identity_map(options);

    if (dout_din)
    {
      dout_din->set(I * dt, var, var_rate);
      if (Model::stage == Model::Stage::UPDATING)
      {
        dout_din->set(I, var, var_n);
        dout_din->set(s_dot, var, time);
        dout_din->set(-s_dot, var, time_n);
      }
    }

    if (d2out_din2)
      if (Model::stage == Model::Stage::UPDATING)
      {
        d2out_din2->set(I, var, var_rate, time);
        d2out_din2->set(-I, var, var_rate, time_n);
        d2out_din2->set(I, var, time, var_rate);
        d2out_din2->set(-I, var, time_n, var_rate);
      }
  }
}

template class TimeIntegration<Scalar>;
template class TimeIntegration<SymR2>;
} // namespace neml2
