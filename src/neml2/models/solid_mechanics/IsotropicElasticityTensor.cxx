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
  neml_assert(_coef_types.size() == 2,
              "IsotropicElasticityTensor requires exactly two input parameters.");
}

void
IsotropicElasticityTensor::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "IsotropicElasticityTensor doesn't implement second derivatives.");

  const auto [lambda, dl_dp] = convert_to_lambda();
  const auto [mu, dm_dp] = convert_to_mu();

  auto Iv = SSR4::identity_vol(options());
  auto Is = SSR4::identity_sym(options());

  if (out)
    _p = 3.0 * lambda * Iv + 2.0 * mu * Is;

  if (dout_din)
  {
    if (const auto * const p = nl_param("c"))
      _p.d(*p) = 3.0 * dl_dp * Tensor(Iv) + 2.0 * dm_dp * Tensor(Is);
  }
}

std::tuple<Scalar, Tensor>
IsotropicElasticityTensor::convert_to_lambda()
{
  Scalar p1 = _coef.index({0});
  Scalar p2 = _coef.index({1});

  ParamType p1_type = _coef_types[0];
  ParamType p2_type = _coef_types[1];

  if ((p1_type == ParamType::YOUNGS) && (p2_type == ParamType::POISSONS))
    return std::make_tuple(
        p1 * p2 / ((1 + p2) * (1 - 2 * p2)),
        math::base_stack(std::vector<Scalar>(
            {-p2 / (2 * p2 * p2 + p2 - 1),
             (p1 + 2 * p1 * p2 * p2) / ((2 * p2 * p2 + p2 - 1) * (2 * p2 * p2 + p2 - 1))})));
  if ((p1_type == ParamType::POISSONS) && (p2_type == ParamType::YOUNGS))
    return std::make_tuple(
        p2 * p1 / ((1 + p1) * (1 - 2 * p1)),
        math::base_stack(std::vector<Scalar>(
            {(p2 + 2 * p2 * p1 * p1) / ((2 * p1 * p1 + p1 - 1) * (2 * p1 * p1 + p1 - 1)),
             -p1 / (2 * p1 * p1 + p1 - 1)})));
  throw NEMLException("Unsupported combination of input parameter types");
}

std::tuple<Scalar, Tensor>
IsotropicElasticityTensor::convert_to_mu()
{
  Scalar p1 = _coef.index({0});
  Scalar p2 = _coef.index({1});

  ParamType p1_type = _coef_types[0];
  ParamType p2_type = _coef_types[1];

  if ((p1_type == ParamType::YOUNGS) && (p2_type == ParamType::POISSONS))
    return std::make_tuple(p1 / (2 * (1 + p2)),
                           math::base_stack(std::vector<Scalar>(
                               {1.0 / (2.0 + 2 * p2), -p1 / (2 * (1 + p2) * (1 + p2))})));
  if ((p1_type == ParamType::POISSONS) && (p2_type == ParamType::YOUNGS))
    return std::make_tuple(
        p2 / (2 * (1 + p1)),
        math::base_stack(std::vector<Scalar>({-p2 / (2 * (1 + p1) * (1 + p1)), 1 / (2 + 2 * p1)})));
  throw NEMLException("Unsupported combination of input parameter types");
}

} // namespace neml2
