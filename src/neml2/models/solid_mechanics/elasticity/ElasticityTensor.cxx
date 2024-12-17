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

#include "neml2/models/solid_mechanics/elasticity/ElasticityTensor.h"

#include "neml2/base/MultiEnumSelection.h"

namespace neml2
{

template <std::size_t N>
OptionSet
ElasticityTensor<N>::expected_options()
{
  OptionSet options = NonlinearParameter<SSR4>::expected_options();
  options.doc() = "Define an elasticity tensor in terms of other parameters.";

  MultiEnumSelection type_selection({"LAME_FIRST_CONSTANT",
                                     "BULK_MODULUS",
                                     "SHEAR_MODULUS",
                                     "YOUNGS_MODULUS",
                                     "POISSONS_RATIO",
                                     "P_WAVE_MODULUS",
                                     "INVALID"},
                                    {static_cast<int>(LameParameter::LAME_FIRST_CONSTANT),
                                     static_cast<int>(LameParameter::BULK_MODULUS),
                                     static_cast<int>(LameParameter::SHEAR_MODULUS),
                                     static_cast<int>(LameParameter::YOUNGS_MODULUS),
                                     static_cast<int>(LameParameter::POISSONS_RATIO),
                                     static_cast<int>(LameParameter::P_WAVE_MODULUS),
                                     static_cast<int>(LameParameter::INVALID)},
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

template <std::size_t N>
ElasticityTensor<N>::ElasticityTensor(const OptionSet & options)
  : NonlinearParameter<SSR4>(options),
    _coefs(options.get<std::vector<CrossRef<Scalar>>>("coefficients")),
    _coef_types(options.get<MultiEnumSelection>("coefficient_types").as<LameParameter>()),
    _coef_as_param(options.get<std::vector<bool>>("coefficient_as_parameter"))
{
  neml_assert(_coefs.size() == N, "Expected ", N, " coefficients, got ", _coefs.size(), ".");
  neml_assert(_coef_types.size() == N,
              "Expected ",
              N,
              " entries in coefficient_types, got ",
              _coef_types.size(),
              ".");
  neml_assert(_coef_as_param.size() == 1 || _coef_as_param.size() == N,
              "Expected 1 or ",
              N,
              " entrie(s) in coefficient_as_parameter, got ",
              _coef_as_param.size(),
              ".");

  if (_coef_as_param.size() == 1)
    _coef_as_param.resize(N, _coef_as_param[0]);

  // Fill out _constants, _constant_types, and _constant_names by sorting the coefficients according
  // to the order defined by LameParameter
  declare_elastic_constant(LameParameter::LAME_FIRST_CONSTANT);
  declare_elastic_constant(LameParameter::BULK_MODULUS);
  declare_elastic_constant(LameParameter::SHEAR_MODULUS);
  declare_elastic_constant(LameParameter::YOUNGS_MODULUS);
  declare_elastic_constant(LameParameter::POISSONS_RATIO);
  declare_elastic_constant(LameParameter::P_WAVE_MODULUS);

  // Figure out which coefficients need derivatives
  for (std::size_t i = 0; i < _constant_types.size(); i++)
    if (nl_param(neml2::name(_constant_types[i])))
      _need_derivs[i] = true;
}

template <std::size_t N>
void
ElasticityTensor<N>::declare_elastic_constant(LameParameter ptype)
{
  for (std::size_t i = 0; i < _coefs.size(); i++)
  {
    if (_coef_types[i] != ptype)
      continue;

    neml_assert(std::find(_constant_types.begin(), _constant_types.end(), ptype) ==
                    _constant_types.end(),
                "Duplicate coefficient type provided for coefficient ",
                i,
                ".");

    const auto pname = neml2::name(ptype);
    const auto * pval = _coef_as_param[i]
                            ? &declare_parameter(pname, _coefs[i], /*allow_nonlinear*/ true)
                            : &declare_buffer(pname, _coefs[i]);

    _constant_types[_counter] = ptype;
    _constants[_counter] = pval;
    _counter++;

    return;
  }
}

template class ElasticityTensor<2>;  // isotropic
template class ElasticityTensor<3>;  // cubic/orthotropic
template class ElasticityTensor<5>;  // hexagonal
template class ElasticityTensor<6>;  // tetragonal/rhombohedral
template class ElasticityTensor<7>;  // tetragonal/rhombohedral
template class ElasticityTensor<9>;  // orthorhombic
template class ElasticityTensor<13>; // monoclinic
template class ElasticityTensor<21>; // triclinic
} // namespace neml2
