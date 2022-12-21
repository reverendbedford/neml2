#include "neml2/models/solid_mechanics/PerfectlyPlasticYieldFunction.h"

namespace neml2
{

PerfectlyPlasticYieldFunction::PerfectlyPlasticYieldFunction(
    const std::string & name, const std::shared_ptr<StressMeasure> & sm, Scalar s0)
  : YieldFunction(name, sm, s0, false, false)
{
}

} // namespace neml2
