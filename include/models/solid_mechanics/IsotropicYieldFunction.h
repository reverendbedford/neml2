#pragma once

#include "models/solid_mechanics/YieldFunction.h"

namespace neml2
{
class IsotropicYieldFunction : public YieldFunction
{
public:
  IsotropicYieldFunction(const std::string & name);

  const LabeledAxisAccessor isotropic_hardening;
};
} // namespace neml2
