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
  params.set<std::vector<std::string>>("rate_variable");
  params.set<std::vector<std::string>>("variable");
  return params;
}

template <typename T>
BackwardEulerTimeIntegration<T>::BackwardEulerTimeIntegration(const ParameterSet & params)
  : Model(params),
    _var_rate_name(params.get<std::vector<std::string>>("rate_variable")),
    _var_name(params.get<std::vector<std::string>>("variable")),
    res(declareOutputVariable<T>("residual", _var_name)),
    var_rate(declareInputVariable<T>("state", _var_rate_name)),
    var(declareInputVariable<T>("state", _var_name)),
    var_n(declareInputVariable<T>("old_state", _var_name)),
    time(declareInputVariable<Scalar>({"forces", "time"})),
    time_n(declareInputVariable<Scalar>({"old_forces", "time"}))
{
  setup();
}

template <typename T>
void
BackwardEulerTimeIntegration<T>::set_value(LabeledVector in,
                                           LabeledVector out,
                                           LabeledMatrix * dout_din) const
{
  TorchSize nbatch = in.batch_size();
  auto dt = in.get<Scalar>(time) - in.get<Scalar>(time_n);

  // r = s_np1 - s_n - s_dot * (t_np1 - t_n)
  auto s_np1 = in(var);
  auto s_n = in(var_n);
  auto s_dot = in(var_rate);
  auto r = s_np1 - s_n - s_dot * dt;
  out.set(r, res);

  if (dout_din)
  {
    auto n_state = output().storage_size(res);
    auto I = BatchTensor<1>::identity(n_state).batch_expand(nbatch);
    dout_din->set(I, res, var);
    dout_din->set(-I * dt, res, var_rate);

    if (Model::stage == Model::Stage::SOLVING)
      return;

    dout_din->set(-I, res, var_n);
    dout_din->set(-s_dot, res, time);
    dout_din->set(s_dot, res, time_n);
  }
}

template class BackwardEulerTimeIntegration<Scalar>;
template class BackwardEulerTimeIntegration<SymR2>;
} // namespace neml2
