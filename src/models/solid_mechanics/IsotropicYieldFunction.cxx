#include "models/solid_mechanics/IsotropicYieldFunction.h"

namespace neml2
{
IsotropicYieldFunction::IsotropicYieldFunction(const std::string & name, Scalar s0)
  : YieldFunction(name, s0),
    isotropic_hardening(declareInputVariable<Scalar>({"state", "hardening_interface", "isotropic_hardening"}))
{
  setup();
}
} // namespace neml2
