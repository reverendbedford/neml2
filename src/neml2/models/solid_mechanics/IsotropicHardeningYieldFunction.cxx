#include "neml2/models/solid_mechanics/IsotropicHardeningYieldFunction.h"

namespace neml2
{

IsotropicHardeningYieldFunction::IsotropicHardeningYieldFunction(
    const std::string & name, const std::shared_ptr<StressMeasure> & sm, Scalar s0)
  : YieldFunction(name, sm, s0, true, false)
{
}

} // namespace neml2
