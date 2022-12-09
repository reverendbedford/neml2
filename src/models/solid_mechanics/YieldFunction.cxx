#include "models/solid_mechanics/YieldFunction.h"

namespace neml2
{
YieldFunction::YieldFunction(const std::string & name)
  : SecDerivModel(name)
{
  input().add<LabeledAxis>("state");
  input().subaxis("state").add<SymR2>("mandel_stress");

  output().add<LabeledAxis>("state");
  output().subaxis("state").add<Scalar>("yield_function");
  setup();
}
} // namespace neml2
