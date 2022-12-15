#pragma once

#include "models/Model.h"

namespace neml2
{
class KinematicHardening : public Model
{
public:
  KinematicHardening(const std::string & name);

  const LabeledAxisAccessor cauchy_stress;
  const LabeledAxisAccessor mandel_stress;
};
} // namespace neml2
