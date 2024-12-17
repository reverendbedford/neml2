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

#include "neml2/models/solid_mechanics/IsotropicElasticityTensor.h"

#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(IsotropicElasticityTensor);

OptionSet
IsotropicElasticityTensor::expected_options()
{
  OptionSet options = ElasticityTensor::expected_options();
  options.doc() += "  This class defines an isotropic elasticity tensor using two parameters."
                   "  Various options are available for which two parameters to provide.";

  return options;
}

IsotropicElasticityTensor::IsotropicElasticityTensor(const OptionSet & options)
  : ElasticityTensor(options)
{
}

void
IsotropicElasticityTensor::diagnose(std::vector<Diagnosis> & diagnostics) const
{
  ElasticityTensor::diagnose(diagnostics);

  diagnostic_assert(diagnostics,
                    _constants.size() == 2,
                    "IsotropicElasticityTensor requires two and only two elastic constants, got ",
                    _constants.size());
}

void
IsotropicElasticityTensor::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "IsotropicElasticityTensor doesn't implement second derivatives.");

  const auto [lambda_and_derivs, mu_and_derivs] = convert();
  const auto [lambda, dlambda_dp1, dlambda_dp2] = lambda_and_derivs;
  const auto [mu, dmu_dp1, dmu_dp2] = mu_and_derivs;

  const auto Iv = SSR4::identity_vol(lambda.options());
  const auto Is = SSR4::identity_sym(mu.options());

  if (out)
    _p = 3.0 * lambda * Iv + 2.0 * mu * Is;

  if (dout_din)
  {
    if (const auto * const p1 = nl_param(_constant_names[0]))
      _p.d(*p1) = 3.0 * dlambda_dp1 * Iv + 2.0 * dmu_dp1 * Is;

    if (const auto * const p2 = nl_param(_constant_names[0]))
      _p.d(*p2) = 3.0 * dlambda_dp2 * Iv + 2.0 * dmu_dp2 * Is;
  }
}

std::pair<IsotropicElasticityTensor::ConversionResult, IsotropicElasticityTensor::ConversionResult>
IsotropicElasticityTensor::convert() const
{
  const auto dispatch_key = std::make_pair(_constant_types[0], _constant_types[1]);
  neml_assert(_converters.count(dispatch_key) == 1,
              "Conversion from ",
              _constant_names[0],
              " and ",
              _constant_names[1],
              " is not supported.");
  auto [lambda, mu] = _converters.at(dispatch_key);
  const auto & c0 = _constants[0];
  const auto & c1 = _constants[1];
  return {lambda(c0, c1), mu(c0, c1)};
}

IsotropicElasticityTensor::ConversionResult
IsotropicElasticityTensor::E_nu_to_lambda(const Scalar & E, const Scalar & nu)
{
  const auto lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  const auto dlambda_dE = -nu / (2 * nu * nu + nu - 1);
  const auto dlambda_dnu =
      (E + 2 * E * nu * nu) / ((2 * nu * nu + nu - 1) * (2 * nu * nu + nu - 1));

  return {lambda, dlambda_dE, dlambda_dnu};
}

IsotropicElasticityTensor::ConversionResult
IsotropicElasticityTensor::E_nu_to_mu(const Scalar & E, const Scalar & nu)
{
  const auto mu = E / (2 * (1 + nu));
  const auto dmu_dE = 1.0 / (2.0 + 2 * nu);
  const auto dmu_dnu = -E / (2 * (1 + nu) * (1 + nu));

  return {mu, dmu_dE, dmu_dnu};
}

} // namespace neml2
