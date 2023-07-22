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
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
register_NEML2_object(ChabochePlasticHardening);

ParameterSet
ChabochePlasticHardening::expected_params()
{
  ParameterSet params = FlowRule::expected_params();
  params.set<LabeledAxisAccessor>("back_stress") = {{"state", "internal", "X"}};
  params.set<LabeledAxisAccessor>("flow_direction") = {{"state", "internal", "NM"}};
  params.set<Real>("C");
  params.set<Real>("g");
  params.set<Real>("A");
  params.set<Real>("a");
  return params;
}

ChabochePlasticHardening::ChabochePlasticHardening(const ParameterSet & params)
  : FlowRule(params),
    back_stress(declare_input_variable<SymR2>(params.get<LabeledAxisAccessor>("back_stress"))),
    flow_direction(
        declare_input_variable<SymR2>(params.get<LabeledAxisAccessor>("flow_direction"))),
    back_stress_rate(declare_output_variable<SymR2>(back_stress.with_suffix("_rate"))),
    _C(register_parameter(name() + "/C", Scalar(params.get<Real>("C")))),
    _g(register_parameter(name() + "/g", Scalar(params.get<Real>("g")))),
    _A(register_parameter(name() + "/A", Scalar(params.get<Real>("A")))),
    _a(register_parameter(name() + "/a", Scalar(params.get<Real>("a"))))
{
  setup();
}

void
ChabochePlasticHardening::set_value(const LabeledVector & in,
                                    LabeledVector * out,
                                    LabeledMatrix * dout_din,
                                    LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "Chaboche model doesn't implement second derivatives.");

  SymR2 X = in.get<SymR2>(back_stress);
  Scalar gamma_dot = in.get<Scalar>(flow_rate);
  SymR2 NM = in.get<SymR2>(flow_direction);

  // The effective stress
  Scalar eff = X.norm(EPS);
  // The part that's proportional to the plastic strain rate
  auto g_term = 2.0 / 3.0 * _C * NM - _g * X;

  if (out)
  {
    // The static recovery term
    auto s_term = -_A * eff.pow(_a - 1) * X;
    auto X_dot = g_term * gamma_dot + s_term;
    out->set(X_dot, back_stress_rate);
  }

  if (dout_din)
  {
    auto I = SymR2::identity_map(in.options());

    dout_din->set(g_term, back_stress_rate, flow_rate);
    dout_din->set(2.0 / 3.0 * _C * gamma_dot * I, back_stress_rate, flow_direction);
    dout_din->set(-_g * gamma_dot * I -
                      _A * eff.pow(_a - 3) * ((_a - 1) * X.outer(X) + eff * eff * I),
                  back_stress_rate,
                  back_stress);
  }
}

} // namespace neml2
