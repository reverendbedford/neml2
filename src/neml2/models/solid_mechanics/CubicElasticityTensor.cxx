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
CubicElasticityTensor::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "CubicElasticityTensor doesn't implement second derivatives.");

  const auto [C1, dC1_dp] = convert_to_C1();
  const auto [C2, dC2_dp] = convert_to_C2();
  const auto [C3, dC3_dp] = convert_to_C3();

  auto I1 = SSR4::identity_C1(options());
  auto I2 = SSR4::identity_C2(options());
  auto I3 = SSR4::identity_C3(options());

  if (out)
    _p = C1 * I1 + C2 * I2 + C3 * I3;

  if (dout_din)
  {
    if (const auto * const p = nl_param("c"))
      _p.d(*p) = dC1_dp * Tensor(I1) + dC2_dp * Tensor(I2) + dC3_dp * Tensor(I3);
  }
}

std::tuple<Scalar, Tensor>
CubicElasticityTensor::convert_to_C1()
{
  Scalar p1 = _coef.index({0});
  Scalar p2 = _coef.index({1});
  Scalar p3 = _coef.index({2});

  ParamType p1_type = _coef_types[0];
  ParamType p2_type = _coef_types[1];
  ParamType p3_type = _coef_types[2];

  if ((p1_type == ParamType::YOUNGS) && (p2_type == ParamType::POISSONS) &&
      (p3_type == ParamType::SHEAR))
    return std::make_tuple(
        p1 / ((1 + p2) * (1 - 2 * p2)) * (1 - p2),
        math::base_stack(std::vector<Scalar>(
            {1.0 / ((1 + p2) * (1 - 2 * p2)) * (1 - p2),
             (-2.0 * (p2 - 2.0) * p2 * p1) / ((2.0 * p2 * p2 + p2 - 1) * (2.0 * p2 * p2 + p2 - 1)),
             Scalar::zeros_like(p1)})));
  throw NEMLException("Unsupported combination of input parameter types: " +
                      std::string(input_options().get<EnumSelection>("p1_type")) + " and " +
                      std::string(input_options().get<EnumSelection>("p2_type")) + " and " +
                      std::string(input_options().get<EnumSelection>("p3_type")));
}

std::tuple<Scalar, Tensor>
CubicElasticityTensor::convert_to_C2()
{
  Scalar p1 = _coef.index({0});
  Scalar p2 = _coef.index({1});
  Scalar p3 = _coef.index({2});

  ParamType p1_type = _coef_types[0];
  ParamType p2_type = _coef_types[1];
  ParamType p3_type = _coef_types[2];

  if ((p1_type == ParamType::YOUNGS) && (p2_type == ParamType::POISSONS) &&
      (p3_type == ParamType::SHEAR))
    return std::make_tuple(
        p1 / ((1 + p2) * (1 - 2 * p2)) * p2,
        math::base_stack(std::vector<Scalar>(
            {1.0 / ((1 + p2) * (1 - 2 * p2)) * p2,
             (2 * p2 * p2 * p1 + p1) / ((2.0 * p2 * p2 + p2 - 1) * (2.0 * p2 * p2 + p2 - 1)),
             Scalar::zeros_like(p1)})));
  throw NEMLException("Unsupported combination of input parameter types: " +
                      std::string(input_options().get<EnumSelection>("p1_type")) + " and " +
                      std::string(input_options().get<EnumSelection>("p2_type")) + " and " +
                      std::string(input_options().get<EnumSelection>("p3_type")));
}

std::tuple<Scalar, Tensor>
CubicElasticityTensor::convert_to_C3()
{
  Scalar p1 = _coef.index({0});
  Scalar p2 = _coef.index({1});
  Scalar p3 = _coef.index({2});

  ParamType p1_type = _coef_types[0];
  ParamType p2_type = _coef_types[1];
  ParamType p3_type = _coef_types[2];

  if ((p1_type == ParamType::YOUNGS) && (p2_type == ParamType::POISSONS) &&
      (p3_type == ParamType::SHEAR))
    return std::make_tuple(
        2.0 * p3,
        math::base_stack(std::vector<Scalar>(
            {Scalar::zeros_like(p3), Scalar::zeros_like(p3), Scalar::full_like(p3, 2.0)})));
  throw NEMLException("Unsupported combination of input parameter types: " +
                      std::string(input_options().get<EnumSelection>("p1_type")) + " and " +
                      std::string(input_options().get<EnumSelection>("p2_type")) + " and " +
                      std::string(input_options().get<EnumSelection>("p3_type")));
}

} // namespace neml2
