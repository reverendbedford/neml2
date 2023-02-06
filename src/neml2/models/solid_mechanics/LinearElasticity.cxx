// Copyright 2023, UChicago Argonne, LLC
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

#include "neml2/models/solid_mechanics/LinearElasticity.h"

namespace neml2
{
register_NEML2_object(CauchyStressFromElasticStrain);
register_NEML2_object(CauchyStressRateFromElasticStrainRate);
register_NEML2_object(ElasticStrainFromCauchyStress);
register_NEML2_object(ElasticStrainRateFromCauchyStressRate);

template <bool rate, ElasticityType etype>
ParameterSet
LinearElasticity<rate, etype>::expected_params()
{
  ParameterSet params = Model::expected_params();
  params.set<Real>("E");
  params.set<Real>("nu");
  return params;
}

template <bool rate, ElasticityType etype>
LinearElasticity<rate, etype>::LinearElasticity(const ParameterSet & params)
  : Model(params),
    from(declareInputVariable<SymR2>({"state", in_name()})),
    to(declareOutputVariable<SymR2>({"state", out_name()})),
    _T(register_parameter("elasticity_tensor", T(params.get<Real>("E"), params.get<Real>("nu"))))
{
  setup();
}

template <bool rate, ElasticityType etype>
SymSymR4
LinearElasticity<rate, etype>::T(Scalar E, Scalar nu) const
{
  if constexpr (etype == ElasticityType::STIFFNESS)
    return SymSymR4::init(SymSymR4::isotropic_E_nu, {E, nu});
  if constexpr (etype == ElasticityType::COMPLIANCE)
    return SymSymR4::init(SymSymR4::isotropic_E_nu, {E, nu}).inverse();
}

template <bool rate, ElasticityType etype>
constexpr std::string
LinearElasticity<rate, etype>::in_name()
{
  if constexpr (rate)
  {
    if constexpr (etype == ElasticityType::STIFFNESS)
      return "elastic_strain_rate";
    if constexpr (etype == ElasticityType::COMPLIANCE)
      return "cauchy_stress_rate";
  }
  else
  {
    if constexpr (etype == ElasticityType::STIFFNESS)
      return "elastic_strain";
    if constexpr (etype == ElasticityType::COMPLIANCE)
      return "cauchy_stress";
  }
}

template <bool rate, ElasticityType etype>
constexpr std::string
LinearElasticity<rate, etype>::out_name()
{
  if constexpr (rate)
  {
    if constexpr (etype == ElasticityType::STIFFNESS)
      return "cauchy_stress_rate";
    if constexpr (etype == ElasticityType::COMPLIANCE)
      return "elastic_strain_rate";
  }
  else
  {
    if constexpr (etype == ElasticityType::STIFFNESS)
      return "cauchy_stress";
    if constexpr (etype == ElasticityType::COMPLIANCE)
      return "elastic_strain";
  }
}

template <bool rate, ElasticityType etype>
void
LinearElasticity<rate, etype>::set_value(LabeledVector in,
                                         LabeledVector out,
                                         LabeledMatrix * dout_din) const
{
  out.set(_T * in.get<SymR2>(from), to);

  if (dout_din)
    dout_din->set(_T.batch_expand(in.batch_size()), to, from);
}

template class LinearElasticity<false, ElasticityType::STIFFNESS>;
template class LinearElasticity<false, ElasticityType::COMPLIANCE>;
template class LinearElasticity<true, ElasticityType::STIFFNESS>;
template class LinearElasticity<true, ElasticityType::COMPLIANCE>;
} // namespace neml2
