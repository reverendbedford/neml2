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

namespace neml2
{
register_NEML2_object(CubicElasticityTensor);

OptionSet
CubicElasticityTensor::expected_options()
{
  OptionSet options = ElasticityTensor::expected_options();
  options.doc() += "  This class defines an isotropic elasticity tensor using two parameters."
                   "  Various options are available for which two parameters to provide.";

  options.set_parameter<CrossRef<Scalar>>("p1");
  options.set("p1").doc() = "First parameter";

  EnumSelection type_selection({"youngs_modulus", "poissons_ratio", "shear_modulus", "INVALID"},
                               {static_cast<int>(CubicElasticityTensor::ParamType::YOUNGS),
                                static_cast<int>(CubicElasticityTensor::ParamType::POISSONS),
                                static_cast<int>(CubicElasticityTensor::ParamType::SHEAR),
                                static_cast<int>(CubicElasticityTensor::ParamType::INVALID)},
                               "INVALID");
  options.set<EnumSelection>("p1_type") = type_selection;
  options.set("p1_type").doc() =
      "First parameter type. Options are: " + type_selection.candidates_str();

  options.set_parameter<CrossRef<Scalar>>("p2");
  options.set("p2").doc() = "Second parameter";

  options.set<EnumSelection>("p2_type") = type_selection;
  options.set("p2_type").doc() =
      "Second parameter type. Options are: " + type_selection.candidates_str();

  options.set_parameter<CrossRef<Scalar>>("p3");
  options.set("p3").doc() = "Third parameter";

  options.set<EnumSelection>("p3_type") = type_selection;
  options.set("p3_type").doc() =
      "Third parameter type. Options are: " + type_selection.candidates_str();

  return options;
}

CubicElasticityTensor::CubicElasticityTensor(const OptionSet & options)
  : ElasticityTensor(options),
    _p1(declare_parameter<Scalar>("p1", "p1", /*allow nonlinear=*/true)),
    _p1_type(options.get<EnumSelection>("p1_type").as<ParamType>()),
    _p2(declare_parameter<Scalar>("p2", "p2", /*allow nonlinear=*/true)),
    _p2_type(options.get<EnumSelection>("p2_type").as<ParamType>()),
    _p3(declare_parameter<Scalar>("p3", "p3", /*allow nonlinear=*/true)),
    _p3_type(options.get<EnumSelection>("p3_type").as<ParamType>())
{
}

void
CubicElasticityTensor::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "CubicElasticityTensor doesn't implement second derivatives.");

  const auto [C1, dC1_dp1, dC1_dp2, dC1_dp3] = convert_to_C1();
  const auto [C2, dC2_dp1, dC2_dp2, dC2_dp3] = convert_to_C2();
  const auto [C3, dC3_dp1, dC3_dp2, dC3_dp3] = convert_to_C3();

  auto I1 = SSR4::identity_C1(options());
  auto I2 = SSR4::identity_C2(options());
  auto I3 = SSR4::identity_C3(options());

  if (out)
    _p = C1 * I1 + C2 * I2 + C3 * I3;

  if (dout_din)
  {
    if (const auto * const p1 = nl_param("p1"))
      _p.d(*p1) = dC1_dp1 * I1 + dC2_dp1 * I2 + dC3_dp1 * I3;

    if (const auto * const p2 = nl_param("p2"))
      _p.d(*p2) = dC1_dp2 * I1 + dC2_dp2 * I2 + dC3_dp2 * I3;

    if (const auto * const p3 = nl_param("p3"))
      _p.d(*p3) = dC1_dp3 * I1 + dC2_dp3 * I2 + dC3_dp3 * I3;
  }
}

std::tuple<Scalar, Scalar, Scalar, Scalar>
CubicElasticityTensor::convert_to_C1()
{
  if ((_p1_type == ParamType::YOUNGS) && (_p2_type == ParamType::POISSONS) &&
      (_p3_type == ParamType::SHEAR))
    return std::make_tuple(_p1 / ((1 + _p2) * (1 - 2 * _p2)) * (1 - _p2),
                           1.0 / ((1 + _p2) * (1 - 2 * _p2)) * (1 - _p2),
                           (-2.0 * (_p2 - 2.0) * _p2 * _p1) /
                               ((2.0 * _p2 * _p2 + _p2 - 1) * (2.0 * _p2 * _p2 + _p2 - 1)),
                           Scalar::zeros_like(_p1));
  else
    throw NEMLException("Unsupported combination of input parameter types: " +
                        std::string(input_options().get<EnumSelection>("p1_type")) + " and " +
                        std::string(input_options().get<EnumSelection>("p2_type")) + " and " +
                        std::string(input_options().get<EnumSelection>("p3_type")));
}

std::tuple<Scalar, Scalar, Scalar, Scalar>
CubicElasticityTensor::convert_to_C2()
{
  if ((_p1_type == ParamType::YOUNGS) && (_p2_type == ParamType::POISSONS) &&
      (_p3_type == ParamType::SHEAR))
    return std::make_tuple(_p1 / ((1 + _p2) * (1 - 2 * _p2)) * _p2,
                           1.0 / ((1 + _p2) * (1 - 2 * _p2)) * _p2,
                           (2 * _p2 * _p2 * _p1 + _p1) /
                               ((2.0 * _p2 * _p2 + _p2 - 1) * (2.0 * _p2 * _p2 + _p2 - 1)),
                           Scalar::zeros_like(_p1));
  else
    throw NEMLException("Unsupported combination of input parameter types: " +
                        std::string(input_options().get<EnumSelection>("p1_type")) + " and " +
                        std::string(input_options().get<EnumSelection>("p2_type")) + " and " +
                        std::string(input_options().get<EnumSelection>("p3_type")));
}

std::tuple<Scalar, Scalar, Scalar, Scalar>
CubicElasticityTensor::convert_to_C3()
{
  if ((_p1_type == ParamType::YOUNGS) && (_p2_type == ParamType::POISSONS) &&
      (_p3_type == ParamType::SHEAR))
    return std::make_tuple(
        2.0 * _p3, Scalar::zeros_like(_p3), Scalar::zeros_like(_p3), Scalar::full_like(_p3, 2.0));
  else
    throw NEMLException("Unsupported combination of input parameter types: " +
                        std::string(input_options().get<EnumSelection>("p1_type")) + " and " +
                        std::string(input_options().get<EnumSelection>("p2_type")) + " and " +
                        std::string(input_options().get<EnumSelection>("p3_type")));
}

} // namespace neml2
