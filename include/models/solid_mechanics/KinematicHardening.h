#pragma once

#include "models/Model.h"

namespace neml2
{
class KinematicHardening : public Model
{
public:
  KinematicHardening(const std::string & name);

  KinematicHardening(InputParameters & params);
};
} // namespace neml2
