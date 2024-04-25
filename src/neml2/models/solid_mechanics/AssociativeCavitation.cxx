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

#include "neml2/models/solid_mechanics/AssociativeCavitation.h"

namespace neml2
{
register_NEML2_object(AssociativeCavitation);

OptionSet
AssociativeCavitation::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<VariableName>("flow_rate") = VariableName("state", "internal", "gamma_rate");
  options.set<VariableName>("cavitation_direction") = VariableName("state", "internal", "Nf");
  options.set<VariableName>("cavitation_rate") = VariableName("state", "internal", "f_rate");
  return options;
}

AssociativeCavitation::AssociativeCavitation(const OptionSet & options)
  : Model(options),
    _gamma_dot(declare_input_variable<Scalar>("flow_rate")),
    _Nf(declare_input_variable<Scalar>("cavitation_direction")),
    _f_dot(declare_output_variable<Scalar>("cavitation_rate"))
{
}

void
AssociativeCavitation::set_value(bool out, bool dout_din, bool d2out_din2)
{
  // For associative flow,
  // f_dot = - gamma_dot * Nf
  //    Nf = dfp/df

  if (out)
    _f_dot = -_gamma_dot * _Nf;

  if (dout_din)
  {
    _f_dot.d(_gamma_dot) = -_Nf;
    _f_dot.d(_Nf) = -_gamma_dot;
  }

  if (d2out_din2)
  {
    // I don't know when this will be useful, but since it's easy...
    auto I = Scalar::identity_map(options());
    _f_dot.d(_gamma_dot, _Nf) = -I;
    _f_dot.d(_Nf, _gamma_dot) = -I;
  }
}
} // namespace neml2
