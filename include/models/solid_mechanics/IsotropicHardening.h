#pragma once

#include "models/Model.h"

namespace neml2
{
class IsotropicHardening : public Model
{
public:
  IsotropicHardening(const std::string & name);

  const LabeledAxisAccessor equivalent_plastic_strain;
  const LabeledAxisAccessor isotropic_hardening;
};
} // namespace neml2
