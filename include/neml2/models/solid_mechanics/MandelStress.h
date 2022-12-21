#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
class MandelStress : public Model
{
public:
  MandelStress(const std::string & name);

  const LabeledAxisAccessor cauchy_stress;
  const LabeledAxisAccessor mandel_stress;
};
} // namespace neml2
