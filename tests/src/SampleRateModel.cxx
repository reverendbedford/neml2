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

#include "SampleRateModel.h"
#include "neml2/tensors/SymSymR4.h"

using namespace neml2;

register_NEML2_object(SampleRateModel);

SampleRateModel::SampleRateModel(const ParameterSet & params)
  : Model(params),
    _a(register_parameter("a", Scalar(-0.01, default_tensor_options), false)),
    _b(register_parameter("b", Scalar(-0.5, default_tensor_options), false)),
    _c(register_parameter("c", Scalar(-0.9, default_tensor_options), false)),
    _foo(declare_input_variable<Scalar>({"state", "foo"})),
    _bar(declare_input_variable<Scalar>({"state", "bar"})),
    _baz(declare_input_variable<SymR2>({"state", "baz"})),
    _temperature(declare_input_variable<Scalar>({"forces", "temperature"})),
    _foo_rate(declare_output_variable<Scalar>({"state", "foo_rate"})),
    _bar_rate(declare_output_variable<Scalar>({"state", "bar_rate"})),
    _baz_rate(declare_output_variable<SymR2>({"state", "baz_rate"}))
{
  setup();
}

void
SampleRateModel::set_value(const LabeledVector & in,
                           LabeledVector * out,
                           LabeledMatrix * dout_din,
                           LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  // Grab the trial states
  auto foo = in.get<Scalar>(_foo);
  auto bar = in.get<Scalar>(_bar);
  auto baz = in.get<SymR2>(_baz);

  // Say the rates depend on temperature, for fun
  auto T = in.get<Scalar>(_temperature);

  // Some made up rates
  auto foo_dot = (foo * foo + bar) * T + baz.tr();
  auto bar_dot = _a * bar + _b * foo + _c * T + baz.tr();
  auto baz_dot = (foo + bar) * baz * (T - 3);

  // Set the output
  if (out)
  {
    out->set(foo_dot, _foo_rate);
    out->set(bar_dot, _bar_rate);
    out->set(baz_dot, _baz_rate);
  }

  if (dout_din)
  {
    const auto options = in.options();

    auto dfoo_dot_dfoo = 2 * foo * T;
    auto dfoo_dot_dbar = T;
    auto dfoo_dot_dbaz = SymR2::identity(options);
    auto dbar_dot_dfoo = _b;
    auto dbar_dot_dbar = _a;
    auto dbar_dot_dbaz = SymR2::identity(options);
    auto dbaz_dot_dfoo = baz * (T - 3);
    auto dbaz_dot_dbar = baz * (T - 3);
    auto dbaz_dot_dbaz = (foo + bar) * (T - 3) * SymR2::identity_map(options);

    dout_din->set(dfoo_dot_dfoo, _foo_rate, _foo);
    dout_din->set(dfoo_dot_dbar, _foo_rate, _bar);
    dout_din->set(dfoo_dot_dbaz, _foo_rate, _baz);
    dout_din->set(dbar_dot_dfoo, _bar_rate, _foo);
    dout_din->set(dbar_dot_dbar, _bar_rate, _bar);
    dout_din->set(dfoo_dot_dbaz, _bar_rate, _baz);
    dout_din->set(dbaz_dot_dfoo, _baz_rate, _foo);
    dout_din->set(dbaz_dot_dbar, _baz_rate, _bar);
    dout_din->set(dbaz_dot_dbaz, _baz_rate, _baz);

    auto dfoo_dot_dT = foo * foo + bar;
    auto dbar_dot_dT = _c;
    auto dbaz_dot_dT = (foo + bar) * baz;

    dout_din->set(dfoo_dot_dT, _foo_rate, _temperature);
    dout_din->set(dbar_dot_dT, _bar_rate, _temperature);
    dout_din->set(dbaz_dot_dT, _baz_rate, _temperature);
  }
}
