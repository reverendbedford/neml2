#include "models/solid_mechanics/HardeningMap.h"

std::string
HardeningMap::conjugate_name(std::string stress_var) const
{
  return "conjugate_" + stress_var;
}
