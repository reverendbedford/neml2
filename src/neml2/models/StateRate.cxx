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
  options.set<LabeledAxisAccessor>("state");
  options.set<LabeledAxisAccessor>("time") = {"t"};
  return options;
}

template <typename T>
StateRate<T>::StateRate(const OptionSet & options)
  : Model(options),
    state(declare_input_variable<T>(options.get<LabeledAxisAccessor>("state").on("state"))),
    state_n(declare_input_variable<T>(options.get<LabeledAxisAccessor>("state").on("old_state"))),
    time(declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("time").on("forces"))),
    time_n(
        declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("time").on("old_forces"))),
    state_rate(declare_output_variable<T>(
        options.get<LabeledAxisAccessor>("state").with_suffix("_rate").on("state")))
{
  this->setup();
}

template <typename T>
void
StateRate<T>::set_value(const LabeledVector & in,
                        LabeledVector * out,
                        LabeledMatrix * dout_din,
                        LabeledTensor3D * d2out_din2) const
{
  const auto options = in.options();

  auto s_np1 = in.get<T>(state);
  auto s_n = in.get<T>(state_n);
  auto t_np1 = in.get<Scalar>(time);
  auto t_n = in.get<Scalar>(time_n);

  auto ds = s_np1 - s_n;
  auto dt = t_np1 - t_n;

  if (out)
  {
    auto s_dot = ds / dt;
    out->set(s_dot, state_rate);
  }

  if (dout_din || d2out_din2)
  {
    auto I = T::identity_map(options);

    if (dout_din)
    {
      dout_din->set(I / dt, state_rate, state);
      dout_din->set(-I / dt, state_rate, state_n);
      dout_din->set(-ds / dt / dt, state_rate, time);
      dout_din->set(ds / dt / dt, state_rate, time_n);
    }

    if (d2out_din2)
    {
      d2out_din2->set(-I / dt / dt, state_rate, state, time);
      d2out_din2->set(I / dt / dt, state_rate, state, time_n);
      d2out_din2->set(I / dt / dt, state_rate, state_n, time);
      d2out_din2->set(-I / dt / dt, state_rate, state_n, time_n);
      d2out_din2->set(-I / dt / dt, state_rate, time, state);
      d2out_din2->set(I / dt / dt, state_rate, time, state_n);
      d2out_din2->set(2 * ds / dt / dt / dt, state_rate, time, time);
      d2out_din2->set(-2 * ds / dt / dt / dt, state_rate, time, time_n);
      d2out_din2->set(I / dt / dt, state_rate, time_n, state);
      d2out_din2->set(-I / dt / dt, state_rate, time_n, state_n);
      d2out_din2->set(-2 * ds / dt / dt / dt, state_rate, time_n, time);
      d2out_din2->set(2 * ds / dt / dt / dt, state_rate, time_n, time_n);
    }
  }
}

template class StateRate<Scalar>;
template class StateRate<SR2>;
} // namespace neml2
