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
  params.set<LabeledAxisAccessor>("force");
  params.set<LabeledAxisAccessor>("time") = LabeledAxisAccessor{{"t"}};
  return params;
}

template <typename T>
ForceRate<T>::ForceRate(const ParameterSet & params)
  : Model(params),
    force(declare_input_variable<T>(params.get<LabeledAxisAccessor>("force").on("forces"))),
    force_n(declare_input_variable<T>(params.get<LabeledAxisAccessor>("force").on("old_forces"))),
    time(declare_input_variable<Scalar>(params.get<LabeledAxisAccessor>("time").on("forces"))),
    time_n(
        declare_input_variable<Scalar>(params.get<LabeledAxisAccessor>("time").on("old_forces"))),
    force_rate(declare_output_variable<T>(
        params.get<LabeledAxisAccessor>("force").with_suffix("_rate").on("forces")))
{
  this->setup();
}

template <typename T>
void
ForceRate<T>::set_value(LabeledVector in,
                        LabeledVector * out,
                        LabeledMatrix * dout_din,
                        LabeledTensor3D * d2out_din2) const
{
  const auto options = in.options();

  auto f_np1 = in.get<T>(force);
  auto f_n = in.get<T>(force_n);
  auto t_np1 = in.get<Scalar>(time);
  auto t_n = in.get<Scalar>(time_n);

  auto df = f_np1 - f_n;
  auto dt = t_np1 - t_n;

  if (out)
  {
    auto f_dot = df / dt;
    out->set(f_dot, force_rate);
  }

  if (dout_din || d2out_din2)
  {
    auto I = T::identity_map(options);

    if (dout_din)
    {
      dout_din->set(I / dt, force_rate, force);
      dout_din->set(-I / dt, force_rate, force_n);
      dout_din->set(-df / dt / dt, force_rate, time);
      dout_din->set(df / dt / dt, force_rate, time_n);
    }

    if (d2out_din2)
    {
      d2out_din2->set(-I / dt / dt, force_rate, force, time);
      d2out_din2->set(I / dt / dt, force_rate, force, time_n);
      d2out_din2->set(I / dt / dt, force_rate, force_n, time);
      d2out_din2->set(-I / dt / dt, force_rate, force_n, time_n);
      d2out_din2->set(-I / dt / dt, force_rate, time, force);
      d2out_din2->set(I / dt / dt, force_rate, time, force_n);
      d2out_din2->set(2 * df / dt / dt / dt, force_rate, time, time);
      d2out_din2->set(-2 * df / dt / dt / dt, force_rate, time, time_n);
      d2out_din2->set(I / dt / dt, force_rate, time_n, force);
      d2out_din2->set(-I / dt / dt, force_rate, time_n, force_n);
      d2out_din2->set(-2 * df / dt / dt / dt, force_rate, time_n, time);
      d2out_din2->set(2 * df / dt / dt / dt, force_rate, time_n, time_n);
    }
  }
}

template class ForceRate<Scalar>;
template class ForceRate<SymR2>;
} // namespace neml2
