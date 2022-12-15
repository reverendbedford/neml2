#include "models/solid_mechanics/IsotropicYieldFunction.h"

namespace neml2
{
IsotropicYieldFunction::IsotropicYieldFunction(const std::string & name)
  : YieldFunction(name),
    isotropic_hardening(declareInputVariable<Scalar>("state", "isotropic_hardening"))
{
  setup();
}
} // namespace neml2
