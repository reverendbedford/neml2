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

#include "neml2/models/BackwardEulerTimeIntegration.h"

namespace neml2
{
register_NEML2_object(ScalarBackwardEulerTimeIntegration);
register_NEML2_object(SymR2BackwardEulerTimeIntegration);

template <typename T>
ParameterSet
BackwardEulerTimeIntegration<T>::expected_params()
{
  ParameterSet params = Model::expected_params();
  params.set<LabeledAxisAccessor>("variable");
  params.set<LabeledAxisAccessor>("time") = {{"t"}};
  return params;
}

template <typename T>
BackwardEulerTimeIntegration<T>::BackwardEulerTimeIntegration(const ParameterSet & params)
  : Model(params),
    _var_name(params.get<LabeledAxisAccessor>("variable")),
    _var_rate_name(_var_name.with_suffix("_rate")),
    res(declare_output_variable<T>(_var_name.on("residual"))),
    var_rate(declare_input_variable<T>(_var_rate_name.on("state"))),
    var(declare_input_variable<T>(_var_name.on("state"))),
    var_n(declare_input_variable<T>(_var_name.on("old_state"))),
    time(declare_input_variable<Scalar>(params.get<LabeledAxisAccessor>("time").on("forces"))),
    time_n(declare_input_variable<Scalar>(params.get<LabeledAxisAccessor>("time").on("old_forces")))
{
  setup();
}

template <typename T>
void
BackwardEulerTimeIntegration<T>::set_value(LabeledVector in,
                                           LabeledVector * out,
                                           LabeledMatrix * dout_din,
                                           LabeledTensor3D * d2out_din2) const
{
  const auto options = in.options();
  TorchSize nbatch = in.batch_size();

  auto s_dot = in(var_rate);
  auto dt = in.get<Scalar>(time) - in.get<Scalar>(time_n);

  if (out)
  {
    auto s_np1 = in(var);
    auto s_n = in(var_n);
    // r = s_np1 - s_n - s_dot * (t_np1 - t_n)
    out->set(s_np1 - s_n - s_dot * dt, res);
  }

  if (dout_din || d2out_din2)
  {
    auto n_state = output().storage_size(res);
    auto I = BatchTensor<1>::identity(n_state, options).batch_expand(nbatch);

    if (dout_din)
    {

      dout_din->set(I, res, var);
      dout_din->set(-I * dt, res, var_rate);
      if (Model::stage == Model::Stage::UPDATING)
      {
        dout_din->set(-I, res, var_n);
        dout_din->set(-s_dot, res, time);
        dout_din->set(s_dot, res, time_n);
      }
    }

    if (d2out_din2)
    {
      d2out_din2->set(-I, res, var_rate, time);
      d2out_din2->set(I, res, var_rate, time_n);
      if (Model::stage == Model::Stage::UPDATING)
      {
        d2out_din2->set(-I, res, time, var_rate);
        d2out_din2->set(I, res, time_n, var_rate);
      }
    }
  }
}

template class BackwardEulerTimeIntegration<Scalar>;
template class BackwardEulerTimeIntegration<SymR2>;
} // namespace neml2
