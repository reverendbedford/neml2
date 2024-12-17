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

  options.set<std::vector<bool>>("coefficient_as_parameter") = {true};
  options.set("coefficient_as_parameter").doc() =
      "Whether to treat the coefficients as (trainable) parameters. Default is true. Setting this "
      "option to false will treat the coefficients as buffers.";

  return options;
}

ElasticityTensor::ElasticityTensor(const OptionSet & options)
  : NonlinearParameter<SSR4>(options),
    _coefs(options.get<std::vector<CrossRef<Scalar>>>("coefficients")),
    _coef_types(options.get<MultiEnumSelection>("coefficient_types").as<ParamType>()),
    _coef_as_param(options.get<std::vector<bool>>("coefficient_as_parameter")),
    _used(_coefs.size(), false)
{
  neml_assert(_coef_types.size() == _coefs.size(),
              "Number of coefficient types (",
              _coef_types.size(),
              ") does not match number of coefficients (",
              _coefs.size(),
              ").");
  neml_assert(_coef_as_param.size() == 1 || _coef_as_param.size() == _coefs.size(),
              "Number of coefficient as parameter flags (",
              _coef_as_param.size(),
              ") does not match number of coefficients (",
              _coefs.size(),
              "). If only one flag is provided, it will be used for all coefficients.");
  if (_coef_as_param.size() == 1)
    _coef_as_param.resize(_coefs.size(), _coef_as_param[0]);

  // Fill out _constants, _constant_types, and _constant_names by sorting the coefficients according
  // to the order defined by ParamType
  declare_elastic_constants();
}

void
ElasticityTensor::declare_elastic_constants()
{
  for (std::size_t i = 0; i < _coefs.size(); i++)
  {
    neml_assert(_coef_types[i] != ParamType::INVALID,
                "Invalid coefficient type provided for coefficient ",
                i,
                ".");

    const auto & ptype = _coef_types[i];

    neml_assert(std::find(_constant_types.begin(), _constant_types.end(), ptype) ==
                    _constant_types.end(),
                "Duplicate coefficient type provided for coefficient ",
                i,
                ".");

    const auto & pname = param_name.at(ptype);
    const auto * pval = _coef_as_param[i]
                            ? &declare_parameter(pname, _coefs[i], /*allow_nonlinear*/ true)
                            : &declare_buffer(pname, _coefs[i]);

    _constant_types.push_back(ptype);
    _constant_names.push_back(pname);
    _constants.push_back(pval);
  }
}

} // namespace neml2
