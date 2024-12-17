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

#include "neml2/models/solid_mechanics/elasticity/IsotropicElasticityConverter.h"

namespace neml2
{
const IsotropicElasticityConverter::ConversionTableType IsotropicElasticityConverter::table = {
    {{LameParameter::YOUNGS_MODULUS, LameParameter::POISSONS_RATIO},
     {&IsotropicElasticityConverter::E_nu_to_lambda, &IsotropicElasticityConverter::E_nu_to_G}}};

IsotropicElasticityConverter::ConversionType
IsotropicElasticityConverter::E_nu_to_lambda(const InputType & input,
                                             const DerivativeFlagType & deriv)
{
  const auto & E = input[0];
  const auto & nu = input[1];

  const auto lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  const auto dlambda_dE = deriv[0] ? -nu / (2 * nu * nu + nu - 1) : Scalar();
  const auto dlambda_dnu =
      deriv[1] ? (E + 2 * E * nu * nu) / ((2 * nu * nu + nu - 1) * (2 * nu * nu + nu - 1))
               : Scalar();

  return {lambda, {dlambda_dE, dlambda_dnu}};
}

IsotropicElasticityConverter::ConversionType
IsotropicElasticityConverter::E_nu_to_G(const InputType & input, const DerivativeFlagType & deriv)
{
  const auto & E = input[0];
  const auto & nu = input[1];

  const auto G = E / (2 * (1 + nu));
  const auto dG_dE = deriv[0] ? 1.0 / (2.0 + 2 * nu) : Scalar();
  const auto dG_dnu = deriv[1] ? -E / (2 * (1 + nu) * (1 + nu)) : Scalar();

  return {G, {dG_dE, dG_dnu}};
}

} // namespace neml2
