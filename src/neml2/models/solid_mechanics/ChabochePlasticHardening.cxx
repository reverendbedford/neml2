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
  ParameterSet params = PlasticHardening::expected_params();
  params.set<Real>("C");
  params.set<Real>("g");
  params.set<Real>("A");
  params.set<Real>("a");
  params.set<std::string>("backstress_suffix") = "";
  return params;
}

ChabochePlasticHardening::ChabochePlasticHardening(const ParameterSet & params)
  : PlasticHardening(params),
    backstress(declareInputVariable<SymR2>(
        {"state", "internal_state", "backstress" + params.get<std::string>("backstress_suffix")})),
    flow_direction(declareInputVariable<SymR2>({"state", "plastic_flow_direction"})),
    backstress_rate(declareOutputVariable<SymR2>(
        {"state",
         "internal_state",
         "backstress" + params.get<std::string>("backstress_suffix") + "_rate"})),
    _C(register_parameter("chaboche_C" + params.get<std::string>("backstress_suffix"),
                          Scalar(params.get<Real>("C")))),
    _g(register_parameter("chaboche_gamma" + params.get<std::string>("backstress_suffix"),
                          Scalar(params.get<Real>("g")))),
    _A(register_parameter("chaboche_recovery_prefactor" +
                              params.get<std::string>("backstress_suffix"),
                          Scalar(params.get<Real>("A")))),
    _a(register_parameter("chaboche_recovery_exponent" +
                              params.get<std::string>("backstress_suffix"),
                          Scalar(params.get<Real>("a"))))
{
  setup();
}

void
ChabochePlasticHardening::set_value(LabeledVector in,
                                    LabeledVector out,
                                    LabeledMatrix * dout_din) const
{
  // Our backstress
  SymR2 X = in.get<SymR2>(backstress);

  // gamma_dot
  Scalar g = in.get<Scalar>(hardening_rate);

  // Current flow direction
  SymR2 n = in.get<SymR2>(flow_direction);

  // Value of the effective stress for recovery
  auto eff = X.norm(); // Should already be deviatoric

  // Finally we can start assembling the model
  // Proportional to plastic strain rate
  auto g_term = 2.0 / 3.0 * _C * n - _g * X;
  // Static recovery
  auto s_term = -_A * eff.pow(_a - 1.0) * X;
  // Sum and set total
  auto total = g_term * g + s_term;
  out.set(total, backstress_rate);

  if (dout_din)
  {
    auto Y = X / (eff + EPS);

    // Plastic strain rate derivative
    dout_din->set(g_term, backstress_rate, hardening_rate);

    // Useful identity...
    auto I = SymSymR4::init(SymSymR4::identity_sym).batch_expand(in.batch_size());

    // Flow direction derivative
    dout_din->set(2.0 / 3.0 * _C * I * g, backstress_rate, flow_direction);

    // Backstress derivative
    dout_din->set(-torch::Tensor(_g * I * g) -
                      torch::Tensor(_A * (_a - 1.0) * (eff + EPS).pow(_a - 2.0) * X.outer(Y)) -
                      torch::Tensor(_A * eff.pow(_a - 1.0) * I),
                  backstress_rate,
                  backstress);
  }
}

} // namespace neml2
