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
register_NEML2_object(ADSampleRateModel);

template <bool is_ad>
SampleRateModelTempl<is_ad>::SampleRateModelTempl(const ParameterSet & params)
  : SampleRateModelBase<is_ad>(params),
    _foo(this->template declareInputVariable<Scalar>({"state", "foo"})),
    _bar(this->template declareInputVariable<Scalar>({"state", "bar"})),
    _baz(this->template declareInputVariable<SymR2>({"state", "baz"})),
    _temperature(this->template declareInputVariable<Scalar>({"forces", "temperature"})),
    _foo_rate(this->template declareOutputVariable<Scalar>({"state", "foo_rate"})),
    _bar_rate(this->template declareOutputVariable<Scalar>({"state", "bar_rate"})),
    _baz_rate(this->template declareOutputVariable<SymR2>({"state", "baz_rate"}))
{
  this->setup();
}

template <bool is_ad>
void
SampleRateModelTempl<is_ad>::set_value(LabeledVector in,
                                       LabeledVector out,
                                       LabeledMatrix * dout_din) const
{
  // Grab the trial states
  auto foo = in.get<Scalar>(_foo);
  auto bar = in.get<Scalar>(_bar);
  auto baz = in.get<SymR2>(_baz);

  // Say the rates depend on temperature, for fun
  auto T = in.get<Scalar>(_temperature);

  // Some made up rates
  auto foo_dot = (foo * foo + bar) * T + baz.tr();
  auto bar_dot = -bar / 100 - 0.5 * foo - 0.9 * T + baz.tr();
  auto baz_dot = (foo + bar) * baz * (T - 3);

  // Set the output
  out.set(foo_dot, _foo_rate);
  out.set(bar_dot, _bar_rate);
  out.set(baz_dot, _baz_rate);

  if constexpr (!is_ad)
    if (dout_din)
    {
      TorchSize nbatch = in.batch_size();
      auto dfoo_dot_dfoo = 2 * foo * T;
      auto dfoo_dot_dbar = T;
      auto dfoo_dot_dbaz = SymR2::identity().batch_expand(nbatch);
      auto dbar_dot_dfoo = Scalar(-0.5, nbatch);
      auto dbar_dot_dbar = Scalar(-0.01, nbatch);
      auto dbar_dot_dbaz = SymR2::identity().batch_expand(nbatch);
      auto dbaz_dot_dfoo = baz * (T - 3);
      auto dbaz_dot_dbar = baz * (T - 3);
      auto dbaz_dot_dbaz = (foo + bar) * (T - 3) * SymR2::identity_map().batch_expand(nbatch);

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
      auto dbar_dot_dT = Scalar(-0.9, nbatch);
      auto dbaz_dot_dT = (foo + bar) * baz;

      dout_din->set(dfoo_dot_dT, _foo_rate, _temperature);
      dout_din->set(dbar_dot_dT, _bar_rate, _temperature);
      dout_din->set(dbaz_dot_dT, _baz_rate, _temperature);
    }
}

template class SampleRateModelTempl<true>;
template class SampleRateModelTempl<false>;
