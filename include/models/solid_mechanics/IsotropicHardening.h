#pragma once

#include "models/Model.h"

namespace neml2
{
class IsotropicHardening : public Model
{
public:
  IsotropicHardening(const std::string & name);

protected:
  const LabeledAxisAccessor _ep_idx;
  const LabeledAxisAccessor _g_idx;
};
} // namespace neml2
