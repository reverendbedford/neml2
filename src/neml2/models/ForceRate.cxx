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

#include "neml2/models/ForceRate.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
register_NEML2_object(ScalarForceRate);
register_NEML2_object(SymR2ForceRate);

template <typename T>
ParameterSet
ForceRate<T>::expected_params()
{
  ParameterSet params = Model::expected_params();
  params.set<std::string>("force");
  return params;
}

template <typename T>
ForceRate<T>::ForceRate(const ParameterSet & params)
  : Model(params),
    force(declareInputVariable<T>({"forces", params.get<std::string>("force")})),
    force_n(declareInputVariable<T>({"old_forces", params.get<std::string>("force")})),
    time(declareInputVariable<Scalar>({"forces", "time"})),
    time_n(declareInputVariable<Scalar>({"old_forces", "time"})),
    force_rate(declareOutputVariable<T>({"forces", params.get<std::string>("force") + "_rate"}))
{
  this->setup();
}

template <typename T>
void
ForceRate<T>::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  auto f_np1 = in.get<T>(force);
  auto f_n = in.get<T>(force_n);
  auto t_np1 = in.get<Scalar>(time);
  auto t_n = in.get<Scalar>(time_n);

  auto df = f_np1 - f_n;
  auto dt = t_np1 - t_n;
  auto f_dot = df / dt;

  out.set(f_dot, force_rate);

  if (dout_din)
  {
    auto df_dot_df_np1 = T::identity_map() / dt;
    auto df_dot_df_n = -T::identity_map() / dt;
    auto df_dot_dt_np1 = -df / dt / dt;
    auto df_dot_dt_n = df / dt / dt;

    dout_din->set(df_dot_df_np1, force_rate, force);
    dout_din->set(df_dot_df_n, force_rate, force_n);
    dout_din->set(df_dot_dt_np1, force_rate, time);
    dout_din->set(df_dot_dt_n, force_rate, time_n);
  }
}

template class ForceRate<Scalar>;
template class ForceRate<SymR2>;
} // namespace neml2
