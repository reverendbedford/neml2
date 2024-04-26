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

#include "neml2/models/solid_mechanics/ThermalEigenstrain.h"

namespace neml2
{
register_NEML2_object(ThermalEigenstrain);

OptionSet
ThermalEigenstrain::expected_options()
{
  OptionSet options = Eigenstrain::expected_options();
  options.set<VariableName>("temperature") = VariableName("forces", "T");
  options.set<CrossRef<Scalar>>("reference_temperature");
  options.set<CrossRef<Scalar>>("CTE");
  return options;
}

ThermalEigenstrain::ThermalEigenstrain(const OptionSet & options)
  : Eigenstrain(options),
    _T(declare_input_variable<Scalar>("temperature")),
    _T0(declare_parameter<Scalar>("T0", "reference_temperature")),
    _alpha(declare_parameter<Scalar>("alpha", "CTE"))
{
}

void
ThermalEigenstrain::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _eg = _alpha * (_T - _T0) * SR2::identity(options());

  if (dout_din)
    _eg.d(_T) = _alpha * SR2::identity(options());

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
