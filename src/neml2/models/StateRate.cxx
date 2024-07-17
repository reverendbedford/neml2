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

#include "neml2/models/StateRate.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(ScalarStateRate);
register_NEML2_object(SR2StateRate);

template <typename T>
OptionSet
StateRate<T>::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Calculate the first order discrete time derivative of a state variable as \\f$ "
                  "\\dot{s} = \\frac{s-s_n}{t-t_n} \\f$, where \\f$ s \\f$ is the state variable, "
                  "and \\f$ t \\f$ is time.";

  options.set_input<VariableName>("state");
  options.set("state").doc() = "The state variable to take time derivative with";

  options.set_input<VariableName>("time") = VariableName("t");
  options.set("time").doc() = "Time";

  return options;
}

template <typename T>
StateRate<T>::StateRate(const OptionSet & options)
  : Model(options),
    _s(declare_input_variable<T>(options.get<VariableName>("state").prepend("state"))),
    _sn(declare_input_variable<T>(options.get<VariableName>("state").prepend("old_state"))),
    _t(declare_input_variable<Scalar>(options.get<VariableName>("time").prepend("forces"))),
    _tn(declare_input_variable<Scalar>(options.get<VariableName>("time").prepend("old_forces"))),
    _ds_dt(declare_output_variable<T>(
        options.get<VariableName>("state").with_suffix("_rate").prepend("state")))
{
}

template <typename T>
void
StateRate<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  auto ds = _s - _sn;
  auto dt = _t - _tn;

  if (out)
    _ds_dt = ds / dt;

  if (dout_din || d2out_din2)
  {
    auto I = T::identity_map(options());

    if (dout_din)
    {
      _ds_dt.d(_s) = I / dt;
      _ds_dt.d(_sn) = -I / dt;
      _ds_dt.d(_t) = -ds / dt / dt;
      _ds_dt.d(_tn) = ds / dt / dt;
    }

    if (d2out_din2)
    {
      _ds_dt.d(_s, _t) = -I / dt / dt;
      _ds_dt.d(_s, _tn) = I / dt / dt;

      _ds_dt.d(_sn, _t) = I / dt / dt;
      _ds_dt.d(_sn, _tn) = -I / dt / dt;

      _ds_dt.d(_t, _s) = -I / dt / dt;
      _ds_dt.d(_t, _sn) = I / dt / dt;
      _ds_dt.d(_t, _t) = 2 * ds / dt / dt / dt;
      _ds_dt.d(_t, _tn) = -2 * ds / dt / dt / dt;

      _ds_dt.d(_tn, _s) = I / dt / dt;
      _ds_dt.d(_tn, _sn) = -I / dt / dt;
      _ds_dt.d(_tn, _t) = -2 * ds / dt / dt / dt;
      _ds_dt.d(_tn, _tn) = 2 * ds / dt / dt / dt;
    }
  }
}

template class StateRate<Scalar>;
template class StateRate<SR2>;
} // namespace neml2
