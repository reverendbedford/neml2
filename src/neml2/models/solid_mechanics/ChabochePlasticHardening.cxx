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

#include "neml2/models/solid_mechanics/ChabochePlasticHardening.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(ChabochePlasticHardening);

OptionSet
ChabochePlasticHardening::expected_options()
{
  OptionSet options = FlowRule::expected_options();
  options.set<VariableName>("back_stress") = {{"state", "internal", "X"}};
  options.set<VariableName>("flow_direction") = {{"state", "internal", "NM"}};
  options.set<CrossRef<Scalar>>("C");
  options.set<CrossRef<Scalar>>("g");
  options.set<CrossRef<Scalar>>("A");
  options.set<CrossRef<Scalar>>("a");
  return options;
}

ChabochePlasticHardening::ChabochePlasticHardening(const OptionSet & options)
  : FlowRule(options),
    _X(declare_input_variable<SR2>(options.get<VariableName>("back_stress"))),
    _NM(declare_input_variable<SR2>(options.get<VariableName>("flow_direction"))),
    _X_dot(declare_output_variable<SR2>(_X.name().with_suffix("_rate"))),
    _C(declare_parameter<Scalar>("C", "C")),
    _g(declare_parameter<Scalar>("g", "g")),
    _A(declare_parameter<Scalar>("A", "A")),
    _a(declare_parameter<Scalar>("a", "a"))
{
}

void
ChabochePlasticHardening::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Chaboche model doesn't implement second derivatives.");

  // The effective stress
  auto s = SR2(_X).norm(EPS);
  // The part that's proportional to the plastic strain rate
  auto g_term = 2.0 / 3.0 * _C * _NM - _g * _X;

  if (out)
  {
    // The static recovery term
    auto s_term = -_A * math::pow(s, _a - 1) * _X;
    _X_dot = g_term * _gamma_dot + s_term;
  }

  if (dout_din)
  {
    auto I = SR2::identity_map(options());

    _X_dot.d(_gamma_dot) = g_term;
    _X_dot.d(_NM) = 2.0 / 3.0 * _C * _gamma_dot * I;
    _X_dot.d(_X) = -_g * _gamma_dot * I -
                   _A * math::pow(s, _a - 3) * ((_a - 1) * SR2(_X).outer(SR2(_X)) + s * s * I);
  }
}

} // namespace neml2
