#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
class KinematicHardening : public Model
{
public:
  KinematicHardening(const std::string & name);

  const LabeledAxisAccessor plastic_strain;
  const LabeledAxisAccessor kinematic_hardening;
};
} // namespace neml2
