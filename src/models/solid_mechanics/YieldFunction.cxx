#include "models/solid_mechanics/YieldFunction.h"

namespace neml2
{
YieldFunction::YieldFunction(const std::string & name, Scalar s0)
  : SecDerivModel(name),
    mandel_stress(declareInputVariable<SymR2>({"state", "mandel_stress"})),
    yield_function(declareOutputVariable<Scalar>({"state", "yield_function"})),
    _s0(register_parameter("yield_stress", s0))
{
  setup();
}
} // namespace neml2
