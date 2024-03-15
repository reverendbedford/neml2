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

#include "PolynomialModel.h"

using namespace neml2;

register_NEML2_object(PolynomialModel);

OptionSet
PolynomialModel::expected_options()
{
  auto options = Model::expected_options();
  options.set<VariableName>("von_mises_stress") = VariableName("state", "s");
  options.set<VariableName>("temperature") = VariableName("forces", "T");
  options.set<VariableName>("internal_state_1") = VariableName("state", "s1");
  options.set<VariableName>("internal_state_2") = VariableName("state", "s2");
  options.set<VariableName>("equivalent_plastic_strain_rate") = VariableName("state", "ep_dot");
  options.set<VariableName>("internal_state_1_rate") = VariableName("state", "s1_dot");
  options.set<VariableName>("internal_state_2_rate") = VariableName("state", "s2_dot");
  options.set<std::vector<Real>>("s_coefficients");
  options.set<std::vector<Real>>("s1_coefficients");
  options.set<std::vector<Real>>("s2_coefficients");
  return options;
}

PolynomialModel::PolynomialModel(const OptionSet & options)
  : Model(options),
    _s(declare_input_variable<Scalar>("von_mises_stress")),
    _T(declare_input_variable<Scalar>("temperature")),
    _s1(declare_input_variable<Scalar>("internal_state_1")),
    _s2(declare_input_variable<Scalar>("internal_state_2")),
    _ep_dot(declare_output_variable<Scalar>("equivalent_plastic_strain_rate")),
    _s1_dot(declare_output_variable<Scalar>("internal_state_1_rate")),
    _s2_dot(declare_output_variable<Scalar>("internal_state_2_rate")),
    _s_coef(options.get<std::vector<Real>>("s_coefficients")),
    _s1_coef(options.get<std::vector<Real>>("s1_coefficients")),
    _s2_coef(options.get<std::vector<Real>>("s2_coefficients"))
{
}

void
PolynomialModel::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _ep_dot = _s_coef[0] + _s_coef[1] * _s + _s_coef[2] * _s * _s;
    _s1_dot = _s1_coef[0] + _s1_coef[1] * _s1 + _s1_coef[2] * _s1 * _s1;
    _s2_dot = _s2_coef[0] + _s2_coef[1] * _s2 + _s2_coef[2] * _s2 * _s2;
  }

  if (dout_din)
  {
    _ep_dot.d(_s) = _s_coef[1] + 2 * _s_coef[2] * _s;
    _s1_dot.d(_s1) = _s1_coef[1] + 2 * _s1_coef[2] * _s1;
    _s2_dot.d(_s2) = _s2_coef[1] + 2 * _s2_coef[2] * _s2;
  }
}
