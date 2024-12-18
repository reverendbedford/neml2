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

#include "neml2/models/solid_mechanics/elasticity/CubicElasticityConverter.h"

namespace neml2
{
const CubicElasticityConverter::ConversionTableType CubicElasticityConverter::table = {
    {{ElasticConstant::SHEAR_MODULUS,
      ElasticConstant::YOUNGS_MODULUS,
      ElasticConstant::POISSONS_RATIO},
     {&CubicElasticityConverter::G_E_nu_to_C1,
      &CubicElasticityConverter::G_E_nu_to_C2,
      &CubicElasticityConverter::G_E_nu_to_C3}},
};

CubicElasticityConverter::ConversionType
CubicElasticityConverter::G_E_nu_to_C1(const InputType & input, const DerivativeFlagType & deriv)
{
  const auto & G = input[0];
  const auto & E = input[1];
  const auto & nu = input[2];

  const auto C1 = E / ((1 + nu) * (1 - 2 * nu)) * (1 - nu);
  const auto dC1_dG = deriv[0] ? Scalar::zeros(G.options()) : Scalar();
  const auto dC1_dE = deriv[1] ? C1 / E : Scalar();
  const auto dC1_dnu = deriv[2] ? (-2.0 * (nu - 2.0) * nu * E) /
                                      ((2.0 * nu * nu + nu - 1) * (2.0 * nu * nu + nu - 1))
                                : Scalar();

  return {C1, {dC1_dG, dC1_dE, dC1_dnu}};
}

CubicElasticityConverter::ConversionType
CubicElasticityConverter::G_E_nu_to_C2(const InputType & input, const DerivativeFlagType & deriv)
{
  const auto & G = input[0];
  const auto & E = input[1];
  const auto & nu = input[2];

  const auto C2 = E / ((1 + nu) * (1 - 2 * nu)) * nu;
  const auto dC2_dG = deriv[0] ? Scalar::zeros(G.options()) : Scalar();
  const auto dC2_dE = deriv[1] ? C2 / E : Scalar();
  const auto dC2_dnu =
      deriv[2] ? (2 * nu * nu * E + E) / ((2.0 * nu * nu + nu - 1) * (2.0 * nu * nu + nu - 1))
               : Scalar();

  return {C2, {dC2_dG, dC2_dE, dC2_dnu}};
}

CubicElasticityConverter::ConversionType
CubicElasticityConverter::G_E_nu_to_C3(const InputType & input, const DerivativeFlagType & deriv)
{
  const auto & G = input[0];
  const auto & E = input[1];
  const auto & nu = input[2];

  const auto C3 = 2.0 * G;
  const auto dC3_dG = deriv[0] ? Scalar::full(2.0, G.options()) : Scalar();
  const auto dC3_dE = deriv[1] ? Scalar::zeros(E.options()) : Scalar();
  const auto dC3_dnu = deriv[2] ? Scalar::zeros(nu.options()) : Scalar();

  return {C3, {dC3_dG, dC3_dE, dC3_dnu}};
}

} // namespace neml2
