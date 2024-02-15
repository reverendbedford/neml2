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
register_NEML2_object(SR2BackwardEulerTimeIntegration);

template <typename T>
OptionSet
BackwardEulerTimeIntegration<T>::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<VariableName>("variable");
  options.set<VariableName>("time") = VariableName("t");
  return options;
}

template <typename T>
BackwardEulerTimeIntegration<T>::BackwardEulerTimeIntegration(const OptionSet & options)
  : Model(options),
    _var_name(options.get<VariableName>("variable")),
    _var_rate_name(_var_name.with_suffix("_rate")),
    _r(declare_output_variable<T>(_var_name.on("residual"))),
    _ds_dt(declare_input_variable<T>(_var_rate_name.on("state"))),
    _s(declare_input_variable<T>(_var_name.on("state"))),
    _sn(declare_input_variable<T>(_var_name.on("old_state"))),
    _t(declare_input_variable<Scalar>(options.get<VariableName>("time").on("forces"))),
    _tn(declare_input_variable<Scalar>(options.get<VariableName>("time").on("old_forces")))
{
}

template <typename T>
void
BackwardEulerTimeIntegration<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _r = _s - _sn - _ds_dt * (_t - _tn);

  if (dout_din || d2out_din2)
  {
    auto I = BatchTensor::identity(T::const_base_storage, options());

    if (dout_din)
    {
      _r.d(_s) = I;
      _r.d(_ds_dt) = -I * (_t - _tn);
      if (Model::stage == Model::Stage::UPDATING)
      {
        _r.d(_sn) = -I;
        _r.d(_t) = -_ds_dt;
        _r.d(_tn) = _ds_dt;
      }
    }

    if (d2out_din2)
    {
      _r.d(_ds_dt, _t) = -I;
      _r.d(_ds_dt, _tn) = I;
      if (Model::stage == Model::Stage::UPDATING)
      {
        _r.d(_t, _ds_dt) = -I;
        _r.d(_tn, _ds_dt) = I;
      }
    }
  }
}

template class BackwardEulerTimeIntegration<Scalar>;
template class BackwardEulerTimeIntegration<SR2>;
} // namespace neml2
