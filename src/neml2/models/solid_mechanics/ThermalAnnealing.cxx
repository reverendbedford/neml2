// Copyright 2024, UChicago Argonne, LLC
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

#include "neml2/models/solid_mechanics/ThermalAnnealing.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(ScalarThermalAnnealing);
register_NEML2_object(SR2ThermalAnnealing);

template <typename T>
OptionSet
ThermalAnnealing<T>::expected_options()
{
  // This is the only way of getting tensor type in a static method like this...
  // Trim 6 chars to remove 'neml2::'
  auto tensor_type = utils::demangle(typeid(T).name()).substr(7);

  OptionSet options = Model::expected_options();
  options.doc() =
      "Thermal annealing recovery for a hardening variable of type " + tensor_type +
      "."
      "For temperatures below \\f$ T_1 \\f$ the model keeps the base model hardenign rate."
      "For temperatures above \\f$T_1\\f$ but below \\f$T_2 \\f$ the model zeros the hardening "
      "rate."
      "For temperatures above \\f$T_2\\f$ the model replaces the hardening rate with "
      "\\f$ \\dot{h} = \\frac{-h}{\\tau} \\f$ where \\f$ \\tau \\f$ is the rate of recovery.";

  options.set_input("base_rate");
  options.set("base_rate").doc() = "Base hardening rate";

  options.set_input("base");
  options.set("base").doc() = "Underlying base hardening variable";

  options.set_output("modified_rate");
  options.set("modified_rate").doc() = "Output for the modified hardening rate.";

  options.set_input("temperature") = VariableName("forces", "T");
  options.set("temperature").doc() = "Temperature";

  options.set_parameter<CrossRef<Scalar>>("T1");
  options.set("T1").doc() = "First stage annealing temperature";

  options.set_parameter<CrossRef<Scalar>>("T2");
  options.set("T2").doc() = "Second stage annealing temperature";

  options.set_parameter<CrossRef<Scalar>>("tau");
  options.set("tau").doc() = "Recovery rate for second stage annealing.";

  return options;
}

template <typename T>
ThermalAnnealing<T>::ThermalAnnealing(const OptionSet & options)
  : Model(options),
    _base_rate(declare_input_variable<T>("base_rate")),
    _base_h(declare_input_variable<T>("base")),
    _modified_rate(declare_output_variable<T>("modified_rate")),
    _T(declare_input_variable<Scalar>("temperature")),
    _T1(declare_parameter<Scalar>("T1", "T1")),
    _T2(declare_parameter<Scalar>("T2", "T2")),
    _tau(declare_parameter<Scalar>("tau", "tau"))
{
}

template <typename T>
void
ThermalAnnealing<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "ThermalAnnealing model doesn't implement second derivatives.");

  if (out)
  {
    auto base_region = Scalar(torch::where(_T < _T1, 1.0, 0.0), batch_dim());
    auto recover_region = Scalar(torch::where(_T >= _T2, 1.0, 0.0), batch_dim());
    _modified_rate = base_region * _base_rate + recover_region * -_base_h / _tau;
  }

  if (dout_din)
  {
    auto I = T::identity_map(options());

    if (_base_rate.is_dependent())
      _modified_rate.d(_base_rate) = Scalar(torch::where(_T < _T1, 1.0, 0.0), batch_dim()) * I;

    if (_base_h.is_dependent())
      _modified_rate.d(_base_h) =
          Scalar(torch::where(_T < _T1, 0.0, torch::where(_T < _T2, 0.0, -1.0 / _tau)),
                 batch_dim()) *
          I;
  }
}

template class ThermalAnnealing<Scalar>;
template class ThermalAnnealing<SR2>;
} // namespace neml2
