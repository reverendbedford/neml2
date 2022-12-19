#include "models/solid_mechanics/IsotropicAndKinematicHardeningYieldFunction.h"

namespace neml2 
{

IsotropicAndKinematicHardeningYieldFunction::IsotropicAndKinematicHardeningYieldFunction(
    const std::string & name, 
    const std::shared_ptr<StressMeasure> & sm,
    Scalar s0) :
      YieldFunction(name, sm, s0, true, true)
{

}

} // namespace neml2
