#pragma once

#include "models/Model.h"

namespace neml2
{
class KinematicHardening : public Model
{
public:
  KinematicHardening(const std::string & name);

protected:
  const LabeledAxisAccessor _cauchy_idx;
  const LabeledAxisAccessor _mandel_idx;
};
} // namespace neml2
