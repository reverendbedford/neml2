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

  options.set_parameter<std::vector<CrossRef<Scalar>>>("coefficients");
  options.set("coefficients").doc() = "Coefficients used to define the elasticity tensor";

  return options;
}

ElasticityTensor::ElasticityTensor(const OptionSet & options)
  : NonlinearParameter<SSR4>(options),
    _coef(declare_parameters<Scalar>("coef", "coefficients", /*allow_nonlinear=*/true)),
    _coef_types(options.get<MultiEnumSelection>("coefficient_types").as<ParamType>())
{
  neml_assert(_coef_types.size() == _coef.size(),
              "Number of coefficient types must match number of coefficients.");
}

} // namespace neml2
