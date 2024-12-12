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

#include "neml2/models/solid_mechanics/ElasticityTensor.h"

#include "neml2/base/MultiEnumSelection.h"
#include "neml2/misc/math.h"

namespace neml2
{

OptionSet
ElasticityTensor::expected_options()
{
  OptionSet options = NonlinearParameter<SSR4>::expected_options();
  options.doc() = "Define an elasticity tensor in terms of other parameters.";

  MultiEnumSelection type_selection(
      {"youngs_modulus", "poissons_ratio", "shear_modulus", "INVALID"},
      {static_cast<int>(ElasticityTensor::ParamType::YOUNGS),
       static_cast<int>(ElasticityTensor::ParamType::POISSONS),
       static_cast<int>(ElasticityTensor::ParamType::SHEAR),
       static_cast<int>(ElasticityTensor::ParamType::INVALID)},
      {"INVALID"});
  options.set<MultiEnumSelection>("coefficient_types") = type_selection;
  options.set("coefficient_types").doc() =
      "Types for each parameter, options are: " + type_selection.candidates_str();

  options.set_parameter<std::vector<CrossRef<Scalar>>>("coefficients") = {CrossRef<Scalar>("1")};
  options.set("coefficients").doc() = "Coefficients used to define the elasticity tensor";

  options.set<bool>("coefficients_as_parameters") = false;
  options.set("coefficients_as_parameters").doc() =
      "By default, the coefficients are declared as buffers. Set this option to true to declare "
      "them as (trainable) parameters.";

  return options;
}

ElasticityTensor::ElasticityTensor(const OptionSet & options)
  : NonlinearParameter<SSR4>(options),
    _coef_as_param(options.get<bool>("coefficients_as_parameters")),
    _coef(_coef_as_param ? declare_parameter<Tensor>("c", make_coef(options))
                         : declare_buffer<Tensor>("c", make_coef(options))),
    _coef_types(options.get<MultiEnumSelection>("coefficient_types").as<ParamType>())
{
  neml_assert(_coef_types.size() == (unsigned int)_coef.sizes()[0],
              "Number of coefficient types must match number of coefficients.");
}

Tensor
ElasticityTensor::make_coef(const OptionSet & options) const
{
  const auto coefs_in = options.get<std::vector<CrossRef<Scalar>>>("coefficients");
  const std::vector<Scalar> coefs(coefs_in.begin(), coefs_in.end());
  return math::base_stack(coefs);
}

} // namespace neml2
