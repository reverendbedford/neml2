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

#include "neml2/models/ArrheniusParameter.h"

namespace neml2
{
register_NEML2_object(ArrheniusParameter);

OptionSet
ArrheniusParameter::expected_options()
{
  OptionSet options = NonlinearParameter<Scalar>::expected_options();
  options.set<CrossRef<Scalar>>("reference_value");
  options.set<CrossRef<Scalar>>("activation_energy");
  options.set<Real>("ideal_gas_constant");
  options.set<VariableName>("temperature") = VariableName("forces", "T");
  return options;
}

ArrheniusParameter::ArrheniusParameter(const OptionSet & options)
  : NonlinearParameter<Scalar>(options),
    _p0(declare_parameter<Scalar>("p0", "reference_value")),
    _Q(declare_parameter<Scalar>("Q", "activation_energy")),
    _R(options.get<Real>("ideal_gas_constant")),
    _T(declare_input_variable<Scalar>("temperature"))
{
}

void
ArrheniusParameter::set_value(bool out, bool dout_din, bool d2out_din2)
{
  const auto p = _p0 * math::exp(-_Q / _R / _T);

  if (out)
    _p = p;

  if (dout_din || d2out_din2)
  {
    const auto dp_dT = p * _Q / _R / _T / _T;

    if (dout_din)
      _p.d(_T) = dp_dT;

    if (d2out_din2)
      _p.d(_T, _T) = dp_dT * _Q / _R / _T / _T - 2 * p * _Q / _R / _T / _T / _T;
  }
}
} // namespace neml2
