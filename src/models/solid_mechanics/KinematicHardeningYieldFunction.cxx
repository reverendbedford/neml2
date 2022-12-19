#include "models/solid_mechanics/KinematicHardeningYieldFunction.h"

namespace neml2 
{

KinematicHardeningYieldFunction::KinematicHardeningYieldFunction(
    const std::string & name, 
    const std::shared_ptr<StressMeasure> & sm,
    Scalar s0) :
      YieldFunction(name, sm, s0, false, true)
{

}

} // namespace neml2
