#include "neml2/models/solid_mechanics/PlasticFlowRate.h"

namespace neml2
{
PlasticFlowRate::PlasticFlowRate(const std::string & name)
  : Model(name),
    yield_function(declareInputVariable<Scalar>({"state", "yield_function"})),
    hardening_rate(declareOutputVariable<Scalar>({"state", "hardening_rate"}))
{
  setup();
}
} // namespace neml2
