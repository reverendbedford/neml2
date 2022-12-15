#include "models/solid_mechanics/LinearElasticity.h"

namespace neml2
{
template <bool rate, ElasticityType etype>
LinearElasticity<rate, etype>::LinearElasticity(const std::string & name, SymSymR4 T)
  : Model(name),
    from(declareInputVariable<SymR2>({"state", in_name()})),
    to(declareOutputVariable<SymR2>({"state", out_name()})),
    _T(register_parameter("elasticity_tensor", T))
{
  setup();
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
