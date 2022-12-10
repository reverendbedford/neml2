#include "models/solid_mechanics/YieldFunction.h"

namespace neml2
{
YieldFunction::YieldFunction(const std::string & name)
  : SecDerivModel(name),
    _mandel_idx(declareVariable<SymR2>(input(), "state", "mandel_stress")),
    _f_idx(declareVariable<Scalar>(output(), "state", "yield_function"))
{
  setup();
}
} // namespace neml2
