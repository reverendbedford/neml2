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
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(ScalarForceRate);
register_NEML2_object(SR2ForceRate);

template <typename T>
OptionSet
ForceRate<T>::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Calculate the first order discrete time derivative of a force variable as \\f$ "
                  "\\dot{f} = \\frac{f-f_n}{t-t_n} \\f$, where \\f$ f \\f$ is the force variable, "
                  "and \\f$ t \\f$ is time.";

  options.set_output<VariableName>("force");
  options.set("force").doc() = "The force variable to take time derivative with";

  options.set_input<VariableName>("time") = {"t"};
  options.set("time").doc() = "Time";

  return options;
}

template <typename T>
ForceRate<T>::ForceRate(const OptionSet & options)
  : Model(options),
    _df_dt(declare_output_variable<T>(
        options.get<VariableName>("force").with_suffix("_rate").on("forces"))),
    _f(declare_input_variable<T>(options.get<VariableName>("force").on("forces"))),
    _fn(declare_input_variable<T>(options.get<VariableName>("force").on("old_forces"))),
    _t(declare_input_variable<Scalar>(options.get<VariableName>("time").on("forces"))),
    _tn(declare_input_variable<Scalar>(options.get<VariableName>("time").on("old_forces")))
{
}

template <typename T>
void
ForceRate<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  auto df = _f - _fn;
  auto dt = _t - _tn;

  if (out)
    _df_dt = df / dt;

  if (dout_din || d2out_din2)
  {
    auto I = T::identity_map(options());

    if (dout_din)
    {
      _df_dt.d(_f) = I / dt;
      _df_dt.d(_fn) = -I / dt;
      _df_dt.d(_t) = -df / dt / dt;
      _df_dt.d(_tn) = df / dt / dt;
    }

    if (d2out_din2)
    {
      _df_dt.d(_f, _t) = -I / dt / dt;
      _df_dt.d(_f, _tn) = I / dt / dt;

      _df_dt.d(_fn, _t) = I / dt / dt;
      _df_dt.d(_fn, _tn) = -I / dt / dt;

      _df_dt.d(_t, _f) = -I / dt / dt;
      _df_dt.d(_t, _fn) = I / dt / dt;
      _df_dt.d(_t, _t) = 2 * df / dt / dt / dt;
      _df_dt.d(_t, _tn) = -2 * df / dt / dt / dt;

      _df_dt.d(_tn, _f) = I / dt / dt;
      _df_dt.d(_tn, _fn) = -I / dt / dt;
      _df_dt.d(_tn, _t) = -2 * df / dt / dt / dt;
      _df_dt.d(_tn, _tn) = 2 * df / dt / dt / dt;
    }
  }
}

template class ForceRate<Scalar>;
template class ForceRate<SR2>;
} // namespace neml2
