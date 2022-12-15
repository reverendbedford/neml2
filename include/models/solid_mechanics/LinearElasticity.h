#pragma once

#include "models/Model.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
enum ElasticityType
{
  STIFFNESS,
  COMPLIANCE
};

template <bool rate, ElasticityType etype>
class LinearElasticity : public Model
{
public:
  LinearElasticity(const std::string & name, SymSymR4 T);

  static constexpr std::string in_name();
  static constexpr std::string out_name();

  const LabeledAxisAccessor from;
  const LabeledAxisAccessor to;

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  /**
  The fourth order transformation tensor. When `etype == STIFFNESS`, this is the stiffness tensor;
  when `etype == COMPLIANCE`, this is the compliance tensor.
  */
  SymSymR4 _T;
};

typedef LinearElasticity<false, ElasticityType::STIFFNESS> CauchyStressFromElasticStrain;
typedef LinearElasticity<false, ElasticityType::COMPLIANCE> ElasticStrainFromCauchyStress;
typedef LinearElasticity<true, ElasticityType::STIFFNESS> CauchyStressRateFromElasticStrainRate;
typedef LinearElasticity<true, ElasticityType::COMPLIANCE> ElasticStrainRateFromCauchyStressRate;
} // namespace neml2
