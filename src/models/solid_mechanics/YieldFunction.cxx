#include "models/solid_mechanics/YieldFunction.h"

namespace neml2
{
YieldFunction::YieldFunction(const std::string & name)
  : SecDerivModel(name),
    mandel_stress(declareInputVariable<SymR2>({"state", "mandel_stress"})),
    yield_function(declareOutputVariable<Scalar>({"state", "yield_function"}))
{
  setup();
}
} // namespace neml2
