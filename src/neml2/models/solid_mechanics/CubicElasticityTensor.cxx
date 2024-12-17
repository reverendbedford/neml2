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

#include "neml2/models/solid_mechanics/CubicElasticityTensor.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(CubicElasticityTensor);

OptionSet
CubicElasticityTensor::expected_options()
{
  OptionSet options = ElasticityTensor::expected_options();
  options.doc() +=
      "  This class defines a cubic anisotropic elasticity tensor using three parameters."
      "  Various options are available for which three parameters to provide.";

  return options;
}

CubicElasticityTensor::CubicElasticityTensor(const OptionSet & options)
  : ElasticityTensor(options)
{
}

void
CubicElasticityTensor::diagnose(std::vector<Diagnosis> & diagnostics) const
{
  ElasticityTensor::diagnose(diagnostics);

  diagnostic_assert(diagnostics,
                    _constants.size() == 3,
                    "CubicElasticityTensor requires three and only three elastic constants, got ",
                    _constants.size());
}

void
CubicElasticityTensor::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "CubicElasticityTensor doesn't implement second derivatives.");

  const auto [C1_and_derivs, C2_and_derivs, C3_and_derivs] = convert();
  const auto [C1, dC1_dp1, dC1_dp2, dC1_dp3] = C1_and_derivs;
  const auto [C2, dC2_dp1, dC2_dp2, dC2_dp3] = C2_and_derivs;
  const auto [C3, dC3_dp1, dC3_dp2, dC3_dp3] = C3_and_derivs;

  const auto I1 = SSR4::identity_C1(C1.options());
  const auto I2 = SSR4::identity_C2(C2.options());
  const auto I3 = SSR4::identity_C3(C3.options());

  if (out)
    _p = C1 * I1 + C2 * I2 + C3 * I3;

  if (dout_din)
  {
    if (const auto * const p1 = nl_param(_constant_names[0]))
      _p.d(*p1) = dC1_dp1 * I1 + dC2_dp1 * I2 + dC3_dp1 * I3;
    if (const auto * const p2 = nl_param(_constant_names[1]))
      _p.d(*p2) = dC1_dp2 * I1 + dC2_dp2 * I2 + dC3_dp2 * I3;
    if (const auto * const p3 = nl_param(_constant_names[2]))
      _p.d(*p3) = dC1_dp3 * I1 + dC2_dp3 * I2 + dC3_dp3 * I3;
  }
}

std::tuple<CubicElasticityTensor::ConversionResult,
           CubicElasticityTensor::ConversionResult,
           CubicElasticityTensor::ConversionResult>
CubicElasticityTensor::convert() const
{
  const auto dispatch_key =
      std::make_pair(_constant_types[0], _constant_types[1], _constant_types[2]);
  neml_assert(_converters.count(dispatch_key) == 1,
              "Conversion from ",
              _constant_names[0],
              ", ",
              _constant_names[1],
              ", and ",
              _constant_names[2],
              " is not supported.");
  auto [C1, C2, C3] = _converters.at(dispatch_key);
  const auto & c0 = _constants[0];
  const auto & c1 = _constants[1];
  const auto & c2 = _constants[2];
  return {C1(c0, c1, c2), C2(c0, c1, c2), C3(c0, c1, c2)};
}

CubicElasticityTensor::ConversionResult
CubicElasticityTensor::E_nu_mu_to_C1(const Scalar & E, const Scalar & nu, const Scalar & mu)
{
  const auto C1 = E / ((1 + nu) * (1 - 2 * nu)) * (1 - nu);
  const auto dC1_dE = 1.0 / ((1 + nu) * (1 - 2 * nu)) * (1 - nu);
  const auto dC1_dnu =
      (-2.0 * (nu - 2.0) * nu * E) / ((2.0 * nu * nu + nu - 1) * (2.0 * nu * nu + nu - 1));
  const auto dC1_dmu = Scalar::zeros(mu.options());
  return {C1, dC1_dE, dC1_dnu, dC1_dmu};
}

CubicElasticityTensor::ConversionResult
CubicElasticityTensor::E_nu_mu_to_C2(const Scalar & E, const Scalar & nu, const Scalar & mu)
{
  const auto C2 = E / ((1 + nu) * (1 - 2 * nu)) * nu;
  const auto dC2_dE = 1.0 / ((1 + nu) * (1 - 2 * nu)) * nu;
  const auto dC2_dnu =
      (2 * nu * nu * E + E) / ((2.0 * nu * nu + nu - 1) * (2.0 * nu * nu + nu - 1));
  const auto dC2_dmu = Scalar::zeros(mu.options());
  return {C2, dC2_dE, dC2_dnu, dC2_dmu};
}

CubicElasticityTensor::ConversionResult
CubicElasticityTensor::E_nu_mu_to_C3(const Scalar & E, const Scalar & nu, const Scalar & mu)
{
  const auto C3 = 2.0 * mu;
  const auto dC3_dE = Scalar::zeros(E.options());
  const auto dC3_dnu = Scalar::zeros(nu.options());
  const auto dC3_dmu = Scalar::full(2.0, mu.options());
  return {C3, dC3_dE, dC3_dnu, dC3_dmu};
}

} // namespace neml2
